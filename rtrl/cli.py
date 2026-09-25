"""Run `rtrl COMMAND --help` for the training and frozen-collection contracts."""

import argparse
import asyncio

from .records import external_output, read_run, write_record
from .replay import replay


def parser():
    p = argparse.ArgumentParser(description="GRPO training and rollout selection audits.")
    sub = p.add_subparsers(dest="command", required=True)
    from .train import parser as train_parser
    sub.add_parser("train", parents=[train_parser()], add_help=False, help="Single-H100 comparative GRPO training")
    for name, size, trials, attempts, help_text in (
        ("collect", 8, 1, 4, "Record complete independent groups; never enforce the research deadline"),
        ("local", 4, 8, 1, "Small frozen-model MPS feasibility bank; batched siblings, no production speed claims"),
    ):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--prompts", required=True, help='External JSONL: {"id": "unique", "prompt": "text", "reference": ...}; extra fields reach the verifier')
        s.add_argument("--model", required=True, help="Hugging Face model ID; resolved to an immutable revision before generation")
        s.add_argument("--revision", default="main")
        s.add_argument("--extend", help="Keep a completed trace's original groups and add --attempts fresh retries per trial into a new file; all other collection settings must match")
        s.add_argument("--reward", required=True, help="module:function; synchronous (prompt_row, generated_text) -> finite reward. Example: rtrl.rewards:exact_match")
        s.add_argument("--group-size", type=int, default=size)
        s.add_argument("--attempts", type=int, default=attempts, help="Independent complete groups available per prompt/trial for fresh retry")
        s.add_argument("--trials", type=int, default=trials)
        s.add_argument("--max-tokens", type=int, default=512)
        s.add_argument("--seed", type=int, default=17)
        s.add_argument("--raw-prompt", action="store_true", help="Skip chat templating; input already contains the model's prompt format")
        s.add_argument("--accept-length", action="store_true", help="Explicitly treat the generation token cap as a task terminal state and score it; otherwise mark truncations unobserved")
    c = sub.choices["collect"]
    c.add_argument("--concurrent-groups", type=int, default=1)
    c.add_argument("--max-model-len", type=int, default=4096)
    c.add_argument("--gpu-memory", type=float, default=0.8, help="vLLM allocation fraction; lower context/concurrency for smaller A100s")
    diagnose = sub.add_parser("diagnose", help="Frozen-policy admission comparison; deadline fixed by separate calibration")
    diagnose.add_argument("--trace", required=True)
    diagnose.add_argument("--calibration", required=True)
    probe = sub.add_parser("probe", help="CPU exact/synthetic controls for selection, sampling noise and retry coverage")
    probe.add_argument("--seed", type=int, default=17)
    probe.add_argument("--replicates", type=int, default=1000)
    for name in ("replay", "audit"):
        s = sub.add_parser(name, help="Replay whole-group admission" if name == "replay" else "Compare fixed-checkpoint GRPO gradients; run after the collector exits")
        s.add_argument("--trace", required=True, help="Complete external collector JSONL")
        s.add_argument("--deadline", type=float, help="Seconds since group dispatch; omit for unlimited; exact boundary is admitted")
        s.add_argument("--clock", choices=["ready_s", "generation_s"], default="ready_s", help="ready_s includes verifier completion")
        if name == "audit":
            s.add_argument("--parameters", default="head", help="head (default), all, or exact parameter-name prefix. A head audit is only a parameter-block diagnostic")
            s.add_argument("--device", choices=["cuda", "mps"], default="cuda", help="Explicit MPS audits require an FP32 local trace; no automatic fallback")
    for name, s in sub.choices.items():
        if name == "train":
            continue
        s.add_argument("--output", required=True, help="New output path outside the Git repository; never overwritten")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "train":
            from .train import run
            run(args)
        elif args.command == "collect":
            from .collect import run
            asyncio.run(run(args))
        elif args.command == "local":
            from .local import run
            run(args)
        elif args.command == "probe":
            from .probe import run
            result = run(args)
            with external_output(args.output) as handle:
                write_record(handle, result)
        else:
            manifest, groups = read_run(args.trace)
            if args.command == "audit":
                from .grpo import run
                result = run(manifest, groups, args)
            elif args.command == "diagnose":
                from .diagnose import diagnose
                calibration_manifest, calibration_groups = read_run(args.calibration)
                result = diagnose(manifest, groups, calibration_manifest, calibration_groups)
                result["calibration_trace_sha256"] = calibration_manifest["trace_sha256"]
            else:
                result = replay(groups, args.deadline, args.clock)
            result["trace"] = str(args.trace)
            result["prompts_sha256"] = manifest["prompts_sha256"]
            result["trace_sha256"] = manifest["trace_sha256"]
            with external_output(args.output) as handle:
                write_record(handle, result)
    except (ValueError, RuntimeError, OSError) as error:
        p.exit(1, f"rtrl: {error}\n")


if __name__ == "__main__":
    main()
