"""Single-H100, serial-group GRPO comparisons with real cancellation."""

import argparse
import hashlib
import json
import platform
import random
import signal
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from records import external_output, read_prompts, write_record
from rewards import gsm8k


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-prompts", required=True)
    p.add_argument("--eval-prompts", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiment-log", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--mode", choices=["baseline", "deadline", "random"], default="baseline")
    p.add_argument("--deadline", type=float)
    p.add_argument("--replacement-rate", type=float)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--groups-per-update", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--project", default="rtrl-timeout-selection")
    p.add_argument("--entity")
    p.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    p.add_argument("--initial-eval", help="Reuse and rescore a matching initial evaluation event file")
    p.add_argument("--resume", help="Checkpoint directory; continue into a NEW output directory")
    return p


def prompt_at(rows, seed, occurrence):
    epoch, slot = divmod(occurrence, len(rows))
    order = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(order)
    return rows[order[slot]]


def sample_seed(seed, occurrence, attempt):
    return (seed * 10000019 + occurrence * 2 + attempt) % (2**63 - 1)


def score(row, sample):
    # Token cap is an explicit terminal failure in every condition and evaluation.
    return gsm8k(row, sample["text"]) if sample["finish_reason"] == "stop" else 0.0


def evaluate(model, tokenizer, rows, tokenized, args):
    import torch
    from transformers import GenerationConfig
    from training_rollout import synchronize
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else eos
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    results = []
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    config = GenerationConfig(max_new_tokens=args.max_tokens, do_sample=False,
                              eos_token_id=eos, pad_token_id=pad, use_cache=True,
                              disable_compile=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for offset in range(0, len(rows), args.eval_batch_size):
            batch = rows[offset:offset + args.eval_batch_size]
            ids = [tokenized[row["id"]] for row in batch]
            width = max(map(len, ids))
            tokens = torch.tensor([[pad] * (width - len(x)) + x for x in ids], device=device)
            mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=device)
            output = model.generate(input_ids=tokens, attention_mask=mask, generation_config=config)
            for row, response in zip(batch, output[:, width:].tolist()):
                end = next((i for i, token in enumerate(response) if token in eos_ids), None)
                if end is not None:
                    response = response[:end + 1]
                sample = {"text": tokenizer.decode(response, skip_special_tokens=True),
                          "finish_reason": "stop" if end is not None else "length"}
                results.append({"id": row["id"], **sample, "reward": score(row, sample),
                                "tokens": len(response)})
    synchronize(device)
    return {"accuracy": sum(r["reward"] for r in results) / len(results),
            "truncation_fraction": sum(r["finish_reason"] == "length" for r in results) / len(results),
            "evaluation_s": time.perf_counter() - started, "results": results}


def run(args):
    import torch
    import wandb
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collect import cuda_device
    from train_update import update
    from training_rollout import retry_group

    if min(args.seconds, args.max_updates, args.groups_per_update, args.max_tokens,
           args.eval_every, args.eval_batch_size, args.learning_rate) <= 0 or args.group_size < 2:
        raise ValueError("Positive budgets and group_size >= 2 required")
    if args.mode == "deadline" and (args.deadline is None or args.deadline <= 0):
        raise ValueError("Deadline condition requires a positive frozen --deadline")
    if args.mode != "deadline" and args.deadline is not None:
        raise ValueError("Only the deadline condition accepts --deadline")
    if args.mode == "random" and (args.replacement_rate is None or not 0 <= args.replacement_rate <= 1):
        raise ValueError("Random condition requires --replacement-rate in [0,1]")
    if args.mode != "random" and args.replacement_rate is not None:
        raise ValueError("Only the random condition accepts --replacement-rate")
    log_path = Path(args.experiment_log).expanduser().resolve()
    prior_log = log_path.read_text()  # Required before every launch, including resumes.
    root = Path(__file__).resolve().parent
    if log_path.is_relative_to(root):
        raise ValueError("Experiment log must remain outside Git")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Commit the code before important training runs")
    rows, train_hash = read_prompts(args.train_prompts)
    heldout, eval_hash = read_prompts(args.eval_prompts)
    if {r["id"] for r in rows} & {r["id"] for r in heldout}:
        raise ValueError("Train and held-out prompt IDs overlap")
    output = Path(args.output_dir).expanduser().resolve()
    gpu = cuda_device()
    if "H100" not in gpu:
        raise ValueError("This comparison is configured for one H100")
    previous = json.loads((Path(args.resume) / "state.json").read_text()) if args.resume else None
    if previous and args.initial_eval:
        raise ValueError("A resumed checkpoint needs its own evaluation")
    revision = previous["config"]["revision"] if previous else HfApi().model_info(args.model, revision=args.revision).sha
    config = {**vars(args), "revision": revision, "code_commit": commit, "gpu": gpu,
              "train_sha256": train_hash, "eval_sha256": eval_hash,
              "experiment_log_sha256": hashlib.sha256(prior_log.encode()).hexdigest(),
              "precision": "FP32 parameters/AdamW state; BF16 forward compute",
              "objective": "on-policy sequence-mean GRPO; no KL; one gradient pass",
              "cache": "within-response KV only; no cross-group reuse",
              "scheduler": "serial groups; batched siblings; cancellation at token boundaries",
              "cap_reward": 0, "sampling": "temperature=1, top_p=1, top_k=0",
              "reward": "GSM8K numeric answer: final boxed number or final number; capped=0",
              "python": platform.python_version(),
              "versions": {p: version(p) for p in ("torch", "transformers", "wandb")}}
    if previous:
        for key in ("model", "revision", "mode", "deadline", "replacement_rate", "seed", "group_size",
                    "groups_per_update", "max_tokens", "learning_rate", "train_sha256", "eval_sha256"):
            if config[key] != previous["config"][key]:
                raise ValueError(f"Resume configuration differs: {key}")
    with external_output(output / "config.json") as handle:
        write_record(handle, config)
    try:
        run = wandb.init(project=args.project, entity=args.entity, config=config, dir=str(output),
                         mode=args.wandb_mode, name=f"{args.mode}-s{args.seed}-{output.name}",
                         group=f"{args.model.split('/')[-1]}-gsm8k", job_type=args.mode)
    except Exception as error:
        with log_path.open("a") as handle:
            handle.write(f"\nW&B launch FAILED: {args.mode}, seed {args.seed}, {commit[:8]}, {type(error).__name__}: {error}. No training executed.\n")
        raise
    run.define_metric("train/seconds")
    run.define_metric("eval/*", step_metric="train/seconds")
    code = wandb.Artifact(f"code-{run.id}", type="code", metadata={"commit": commit})
    for name in subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines():
        code.add_file(str(root / name), name=name)
    run.log_artifact(code)
    def journal(message):
        with log_path.open("a") as handle:
            handle.write(f"\n| {time.strftime('%Y-%m-%d %H:%M:%S')} | {args.mode}, seed {args.seed}, {commit[:8]}, {args.seconds}s | {run.url} | {message} | Inspect paired evidence before next decision. |\n")
    journal(f"STARTED; W&B {args.wandb_mode}, run ID {run.id}; offline runs require sync before instance termination.")
    started = time.perf_counter()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    old_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        torch.manual_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
        tokenized = {row["id"]: tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True,
            return_dict=False)
            for row in rows + heldout}
        model = AutoModelForCausalLM.from_pretrained(args.resume or args.model,
            **({} if args.resume else {"revision": revision}), dtype=torch.float32,
            attn_implementation="sdpa").to("cuda").eval()
        if max(map(len, tokenized.values())) + args.max_tokens > model.config.max_position_embeddings:
            raise ValueError("Prompt plus response exceeds model context")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        state = {"batches": 0, "optimizer_steps": 0, "groups": 0, "replacements": 0,
                 "generated_tokens": 0, "discarded_tokens": 0, "padded_decode_tokens": 0,
                 "training_s": 0.0, "evaluation_s": 0.0, "checkpoint_s": 0.0}
        if previous:
            state.update(previous["state"])
            optimizer.load_state_dict(torch.load(Path(args.resume) / "optimizer.pt", weights_only=True))
        with external_output(output / "events.jsonl") as events:
            write_record(events, {"type": "manifest", **config, "run_url": run.url})
            def evaluate_now(initial=False):
                if initial and args.initial_eval:
                    from eval_cache import load_initial_evaluation
                    result = load_initial_evaluation(args.initial_eval, config, heldout, score)
                else:
                    result = evaluate(model, tokenizer, heldout, tokenized, args)
                state["evaluation_s"] += result["evaluation_s"]
                write_record(events, {"type": "eval", **state, **result})
                run.log({"train/seconds": state["training_s"], "train/optimizer_steps": state["optimizer_steps"],
                         **{f"eval/{key}": value for key, value in result.items() if key != "results"}})
                return result["accuracy"]

            def checkpoint(label):
                checkpoint_start = time.perf_counter()
                path = output / f"checkpoint-{label}"
                path.mkdir()
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                torch.save(optimizer.state_dict(), path / "optimizer.pt")
                state["checkpoint_s"] += time.perf_counter() - checkpoint_start
                (path / "state.json").write_text(json.dumps({"config": config, "state": state}, indent=2))
                write_record(events, {"type": "checkpoint", "path": str(path), **state})
                # A reference is explicit; local files must be archived before host termination.
                run.summary[f"checkpoint_{label}"] = f"{subprocess.check_output(['hostname'], text=True).strip()}:{path}"
                return path

            initial_accuracy = evaluate_now(initial=True)
            initial_elapsed = time.perf_counter() - started
            next_eval = (state["optimizer_steps"] // args.eval_every + 1) * args.eval_every
            group_stats = []
            last_eval_step = state["optimizer_steps"]
            last_checkpoint, saved_step = None, None
            final_accuracy = initial_accuracy
            while state["training_s"] < args.seconds and state["optimizer_steps"] < args.max_updates:
                batch_start = time.perf_counter()
                groups = []
                batch_generation = 0.0
                for _ in range(args.groups_per_update):
                    occurrence = state["groups"]
                    row = prompt_at(rows, args.seed, occurrence)
                    seed = sample_seed(args.seed, occurrence, 0)
                    replace = (args.mode == "random" and
                               random.Random(seed + 8675309).random() < args.replacement_rate)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = retry_group(model, tokenizer, tokenized[row["id"]],
                            group_size=args.group_size, max_tokens=args.max_tokens, seed=seed,
                            replacement_seed=sample_seed(args.seed, occurrence, 1),
                            deadline_s=args.deadline, random_replace=replace)
                    accepted = result["accepted"]
                    for attempt in result["attempts"]:
                        for sample in attempt["samples"]:
                            sample["reward"] = score(row, sample) if sample["finish_reason"] != "deadline" else None
                    groups.append({"prompt_token_ids": tokenized[row["id"]], "samples": accepted["samples"]})
                    state["groups"] += 1
                    state["replacements"] += result["replaced"]
                    for field in ("generated_tokens", "discarded_tokens", "padded_decode_tokens"):
                        state[field] += result[field]
                    batch_generation += result["generation_s"]
                    write_record(events, {"type": "group", "occurrence": occurrence, "prompt_id": row["id"],
                        "batch": state["batches"], "optimizer_steps": state["optimizer_steps"], **result})
                    group_stats.append([result["attempts"][0]["generation_s"],
                        sum(s["reward"] for s in accepted["samples"]) / args.group_size,
                        result["discarded_tokens"], int(result["replaced"])])
                diagnostics = update(model, optimizer, groups, "cuda")
                state["batches"] += 1
                state["optimizer_steps"] += diagnostics["optimizer_stepped"]
                state["training_s"] += time.perf_counter() - batch_start
                record = {"type": "update", **state, **{f"batch_{k}": v for k, v in diagnostics.items()},
                          "batch_generation_s": batch_generation,
                          "budget_overrun_s": max(0.0, state["training_s"] - args.seconds)}
                write_record(events, record)
                run.log({"train/seconds": state["training_s"],
                         **{f"train/{k}": v for k, v in record.items() if k != "type"},
                         "train/replacement_fraction": state["replacements"] / state["groups"]})
                print(json.dumps(record), flush=True)
                if state["optimizer_steps"] >= next_eval:
                    final_accuracy = evaluate_now()
                    last_eval_step = state["optimizer_steps"]
                    last_checkpoint = checkpoint(f"step-{last_eval_step}")
                    saved_step = last_eval_step
                    next_eval += args.eval_every
            if state["optimizer_steps"] != last_eval_step:
                final_accuracy = evaluate_now()
            final_checkpoint = last_checkpoint if saved_step == state["optimizer_steps"] else checkpoint("final")
            (final_checkpoint / "state.json").write_text(json.dumps({"config": config, "state": state}, indent=2))
            if group_stats:
                table = wandb.Table(columns=["original_seconds", "accepted_reward", "discarded_tokens", "replaced"], data=group_stats)
                run.log({"diagnostics/group_time_reward": wandb.plot.scatter(table, "original_seconds", "accepted_reward"),
                         "diagnostics/original_seconds": wandb.Histogram([r[0] for r in group_stats])})
            summary = {**state, "initial_accuracy": initial_accuracy, "final_accuracy": final_accuracy,
                       "initialization_and_initial_eval_s": initial_elapsed,
                       "wall_s": time.perf_counter() - started, "run_url": run.url,
                       "checkpoint": str(final_checkpoint),
                       "budget_overrun_s": max(0.0, state["training_s"] - args.seconds)}
            write_record(events, {"type": "end", **summary})
            with external_output(output / "summary.json") as handle:
                write_record(handle, summary)
            artifact = wandb.Artifact(f"training-records-{run.id}", type="training-records", metadata={"code_commit": commit})
            for name in ("config.json", "summary.json", "events.jsonl"):
                artifact.add_file(str(output / name))
            run.log_artifact(artifact)
            run.summary.update(summary)
            journal(f"FINISHED; accuracy {initial_accuracy:.4f} -> {final_accuracy:.4f}; {state['optimizer_steps']} optimizer steps; training {state['training_s']:.1f}s; replacements {state['replacements']}/{state['groups']}. Checkpoint reference in W&B; archive before termination.")
    except BaseException as error:
        journal(f"FAILED: {type(error).__name__}: {error}")
        run.finish(exit_code=1)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    run.finish()


if __name__ == "__main__":
    run(parser().parse_args())
