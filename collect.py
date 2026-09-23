"""Complete frozen-policy groups on one CUDA GPU. No administrative cancellation."""

import asyncio
import time
from importlib.metadata import version

from records import external_output, read_prompts, write_record
from rollout import PROTOCOL_SHA256, check_extension, load_reward, prepare_bank, score_sample, tokenize_prompts


def cuda_device():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES; collection/audit never uses MPS or CPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required (H100 or A100)")
    return torch.cuda.get_device_name(0)


class VLLMBackend:
    def __init__(self, args, revision):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        self.max_tokens = args.max_tokens
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=args.model, revision=revision, tokenizer_revision=revision,
            dtype="bfloat16", tensor_parallel_size=1, seed=args.seed,
            max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory,
            max_num_seqs=args.group_size * args.concurrent_groups,
            enable_prefix_caching=False, generation_config="vllm",
        ))

    async def generate(self, prompt_ids, seed, request_id, started, max_tokens=None):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        params = SamplingParams(
            temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0,
            repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
            max_tokens=max_tokens or self.max_tokens, seed=seed, logprobs=0,
            output_kind=RequestOutputKind.DELTA,
        )
        tokens, logprobs, text, first_token = [], [], [], None
        finish_reason = None
        async for output in self.engine.generate(
            prompt={"prompt_token_ids": prompt_ids}, sampling_params=params, request_id=request_id,
        ):
            part = output.outputs[0]
            if part.token_ids and first_token is None:
                first_token = time.perf_counter() - started
            tokens.extend(part.token_ids)
            logprobs.extend(probs[token].logprob for token, probs in zip(part.token_ids, part.logprobs or []))
            text.append(part.text)
            if output.finished:
                finish_reason = part.finish_reason
        if finish_reason not in ("stop", "length") or len(tokens) != len(logprobs) or not tokens:
            raise RuntimeError("Incomplete generation or missing sampled-token log probabilities")
        return {"token_ids": tokens, "logprobs": logprobs, "text": "".join(text),
                "first_token_s": first_token, "generation_s": time.perf_counter() - started,
                "finish_reason": finish_reason}


async def collect_group(backend, row, prompt_ids, trial, attempt, index, args, reward):
    started = time.perf_counter()
    gid = f"group-{index}"

    async def sample(slot):
        seed = args.seed + index * args.group_size + slot
        result = {"token_ids": [], "logprobs": [], "text": "", "first_token_s": None,
                  "generation_s": 0.0, "finish_reason": None, "reward": None, "seed": seed}
        try:
            result.update(await backend.generate(prompt_ids, seed, f"{gid}-{slot}", started))
            await asyncio.to_thread(score_sample, result, row, reward, args.accept_length)
        except Exception as error:
            result.update(status="error", reward=None, error=f"{type(error).__name__}: {error}")
            if not result["generation_s"]:
                result["generation_s"] = time.perf_counter() - started
        result["ready_s"] = time.perf_counter() - started
        return result

    samples = await asyncio.gather(*(sample(i) for i in range(args.group_size)))
    return {"type": "group", "id": gid, "prompt_id": row["id"], "trial": trial,
            "attempt": attempt, "prompt_token_ids": prompt_ids, "samples": samples}


async def run(args):
    if min(args.attempts, args.trials, args.concurrent_groups, args.max_tokens) < 1 or args.group_size < 2:
        raise ValueError("Positive counts and group_size >= 2 required")
    if args.seed < 0 or not 0 < args.gpu_memory < 1 or args.max_tokens >= args.max_model_len:
        raise ValueError("Invalid seed, GPU memory fraction, or context/token limit")
    rows, input_hash = read_prompts(args.prompts)
    reward, reward_hash = load_reward(args.reward)
    gpu = cuda_device()
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    tokenized = tokenize_prompts(tokenizer, rows, args.max_tokens, args.max_model_len, args.raw_prompt)
    previous, old_groups, jobs = prepare_bank(args, rows, tokenized)
    if args.seed + (len(old_groups) + len(jobs)) * args.group_size >= 2**31:
        raise ValueError("Request seeds would exceed the supported range")
    manifest = {"type": "manifest", "schema": 1, "model": args.model, "revision": revision,
                "dtype": "bfloat16", "gpu": gpu, "seed": args.seed,
                "group_size": args.group_size, "trials": args.trials, "attempts": (previous["attempts"] if previous else 0) + args.attempts,
                "expected_groups": len(old_groups) + len(jobs), "prompt_ids": [r["id"] for r in rows],
                "prompts_sha256": input_hash, "reward": args.reward,
                "reward_source_sha256": reward_hash, "protocol_source_sha256": PROTOCOL_SHA256,
                "backend": "vllm", "seed_scope": "Independent RNG seed per response",
                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": args.max_tokens},
                "max_model_len": args.max_model_len, "concurrent_groups": args.concurrent_groups,
                "gpu_memory": args.gpu_memory, "prefix_caching": False, "raw_prompt": args.raw_prompt,
                "accept_length": args.accept_length, "timing": "Monotonic observation since group dispatch; ready_s includes verifier latency",
                "versions": {p: version(p) for p in ("torch", "vllm", "transformers")}}
    check_extension(manifest, previous)
    with external_output(args.output) as handle:
        write_record(handle, manifest)
        for group in old_groups:
            write_record(handle, group)
        backend = VLLMBackend(args, revision)
        try:
            start = time.perf_counter()
            await asyncio.gather(*(backend.generate(tokenized[rows[0]["id"]], args.seed+i, f"warmup-{i}", start, 1) for i in range(args.group_size)))
            queue = asyncio.Queue()
            for index, job in enumerate(jobs, start=len(old_groups)):
                queue.put_nowait((index, job))

            async def worker():
                while not queue.empty():
                    index, (row, trial, attempt) = queue.get_nowait()
                    group = await collect_group(backend, row, tokenized[row["id"]], trial, attempt, index, args, reward)
                    write_record(handle, group)
            await asyncio.gather(*(worker() for _ in range(args.concurrent_groups)))
            write_record(handle, {"type": "end", "groups": manifest["expected_groups"]})
        finally:
            backend.engine.shutdown()
