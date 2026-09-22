"""Small MPS feasibility runs. Batched siblings; no production speed claims."""

import hashlib
import importlib
import inspect
import math
import random
import time
from importlib.metadata import version
from pathlib import Path

import torch

from records import external_output, read_prompts, write_record


def generate_group(model, tokenizer, prompt, size, seed, max_tokens):
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos:
        raise ValueError("Local collector requires explicit EOS tokens")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt] * size, device=device)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    finished, first = {}, []

    class ObserveEOS(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            # CPU observation synchronizes each decoding step before timing it.
            latest = input_ids[:, -1].tolist()
            elapsed = time.perf_counter() - started
            if not first:
                first.append(elapsed)
            for slot, token in enumerate(latest):
                if token in eos and slot not in finished:
                    finished[slot] = elapsed
            return torch.zeros(size, dtype=torch.bool, device=device)

    config = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, max_new_tokens=max_tokens,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos[0],
        use_cache=True, return_dict_in_generate=True, output_scores=True,
    )
    with torch.inference_mode():
        output = model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                generation_config=config,
                                stopping_criteria=StoppingCriteriaList([ObserveEOS()]))
    sequences = output.sequences[:, len(prompt):].tolist()
    elapsed = time.perf_counter() - started
    lengths = [next((i+1 for i, token in enumerate(seq) if token in eos), len(seq))
               for seq in sequences]
    logprobs = [[] for _ in range(size)]
    for step, score in enumerate(output.scores):
        sampled = output.sequences[:, len(prompt)+step, None]
        values = score.float().log_softmax(-1).gather(-1, sampled).flatten().tolist()
        for slot, value in enumerate(values):
            if step < lengths[slot]:
                logprobs[slot].append(value)
    return [{"token_ids": seq[:length], "logprobs": probs,
             "text": tokenizer.decode(seq[:length], skip_special_tokens=True),
             "first_token_s": first[0], "generation_s": finished.get(slot, elapsed),
             "finish_reason": "stop" if slot in finished else "length"}
            for slot, (seq, length, probs) in enumerate(zip(sequences, lengths, logprobs))], started


def run(args):
    if not torch.backends.mps.is_available():
        raise RuntimeError("The local command explicitly requires MPS")
    if min(args.trials, args.attempts, args.max_tokens) < 1 or args.group_size < 2 or args.seed < 0:
        raise ValueError("Positive counts, nonnegative seed and group_size >= 2 required")
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows, digest = read_prompts(args.prompts)
    module, function = args.reward.split(":")
    reward = getattr(importlib.import_module(module), function)
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    prompts = {row["id"]: tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True, return_dict=False) for row in rows}
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=torch.float32, attn_implementation="sdpa").to("mps").eval()
    model.requires_grad_(False)
    if any(len(p)+args.max_tokens > model.config.max_position_embeddings for p in prompts.values()):
        raise ValueError("Prompt exceeds the model context budget")
    jobs = [(row, trial, attempt) for row in rows for trial in range(args.trials)
            for attempt in range(args.attempts)]
    random.Random(args.seed).shuffle(jobs)
    manifest = {"type": "manifest", "schema": 1, "model": args.model, "revision": revision,
                "backend": "transformers-mps", "dtype": "float32", "seed": args.seed,
                "group_size": args.group_size, "trials": args.trials, "attempts": args.attempts,
                "expected_groups": len(jobs), "prompt_ids": [r["id"] for r in rows],
                "prompts_sha256": digest, "reward": args.reward,
                "reward_source_sha256": hashlib.sha256(Path(inspect.getsourcefile(reward)).read_bytes()).hexdigest(),
                "collector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_tokens": args.max_tokens},
                "accept_length": False, "concurrent_groups": 1, "prefix_caching": False,
                "timing": "Batched siblings; synchronized EOS observations. ready_s also includes batch return/scoring. MPS timings are not production throughput.",
                "seed_scope": "One RNG seed per independently generated group; slots share the RNG stream",
                "versions": {p: version(p) for p in ("torch", "transformers")}}
    with external_output(args.output) as handle:
        write_record(handle, manifest)
        generate_group(model, tokenizer, prompts[rows[0]["id"]], args.group_size, args.seed, 2)
        for index, (row, trial, attempt) in enumerate(jobs):
            samples, started = generate_group(model, tokenizer, prompts[row["id"]], args.group_size,
                                              args.seed+index+1, args.max_tokens)
            for sample in samples:
                sample.update(status="truncated", reward=None)
                if sample["finish_reason"] == "stop":
                    value = float(reward(row, sample["text"]))
                    if not math.isfinite(value):
                        raise ValueError("Verifier returned a nonfinite reward")
                    sample.update(status="complete", reward=value)
                sample["ready_s"] = time.perf_counter()-started
            write_record(handle, {"type": "group", "id": f"group-{index}", "prompt_id": row["id"],
                                   "trial": trial, "attempt": attempt, "seed": args.seed+index+1,
                                   "prompt_token_ids": prompts[row["id"]], "samples": samples})
            print(f"local: {index+1}/{len(jobs)} groups", flush=True)
        write_record(handle, {"type": "end", "groups": len(jobs)})
