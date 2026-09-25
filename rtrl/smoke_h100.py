"""Tiny CUDA collection + gradient execution check; no training or effect estimate."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .records import external_output, read_run, write_record
from .replay import complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="New directory outside Git; downloads a pinned 0.5B model")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        parser.error("Use a new output directory; existing smoke results are never overwritten")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from .collect import cuda_device
    gpu = cuda_device()
    prompts = output / "prompts.jsonl"
    with external_output(prompts) as handle:
        for pid, question, answer in (("smoke-add", "19 + 23", 42), ("smoke-multiply", "37 * 24", 888)):
            write_record(handle, {"id": pid, "prompt": f"Calculate {question}. Show your work briefly, then write Final: <integer>.",
                                  "reference": answer})
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                cwd=Path(__file__).resolve().parent, text=True).strip())
    with external_output(output / "environment.txt") as handle:
        handle.write(f"GPU: {gpu}\n")
        for command in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"],
                        ["nvidia-smi"], [sys.executable, "-m", "pip", "freeze"]):
            handle.write(subprocess.check_output(command, cwd=root, text=True) + "\n")
    trace, audit = output / "trace.jsonl", output / "audit.jsonl"
    cli = [sys.executable, "-m", "rtrl"]
    # Separate processes release the inference engine's GPU allocation before scoring.
    subprocess.run(cli + ["collect", "--prompts", str(prompts),
                          "--model", "Qwen/Qwen2.5-0.5B-Instruct",
                          "--revision", "7ae557604adf67be50417f59c2c2f167def9a775",
                          "--reward", "rtrl.rewards:arithmetic", "--group-size", "4", "--trials", "2", "--attempts", "1",
                          "--max-tokens", "512", "--max-model-len", "2048", "--gpu-memory", "0.5",
                          "--output", str(trace)], cwd=root, check=True)
    manifest, groups = read_run(trace)
    if not all(complete(group) for group in groups):
        raise RuntimeError(f"Incomplete outcomes: inspect {trace}; do not treat truncations as wrong answers")
    subprocess.run(cli + ["audit", "--trace", str(trace), "--device", "cuda",
                          "--parameters", "model.layers.23.mlp.down_proj.weight", "--output", str(audit)],
                   cwd=root, check=True)
    report = json.loads(audit.read_text())
    summary = {"scope": "Execution check only. Inspect inference/scorer discrepancy before research runs; no learning or speedup claim.",
               "trace_sha256": manifest["trace_sha256"], "groups": len(groups),
               "responses": sum(len(g["samples"]) for g in groups),
               "mixed_reward_groups": sum(len({s["reward"] for s in g["samples"]}) > 1 for g in groups),
               "gradient_norm": report["gradient"]["baseline_norm"],
               "inference_vs_scorer": report["inference_vs_scorer"]["baseline"]}
    with external_output(output / "summary.json") as handle:
        write_record(handle, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
