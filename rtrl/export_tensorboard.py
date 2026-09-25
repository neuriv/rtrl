"""Export immutable TensorBoard snapshots from completed or active training runs."""

import argparse
import hashlib
import json
from pathlib import Path

from .records import external_output


def read_events(directory):
    path = Path(directory).expanduser().resolve() / "events.jsonl"
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events, partial = [], False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Malformed event at {path}:{i + 1}") from None
            partial = True
    if not events or events[0].get("type") != "manifest":
        raise ValueError(f"Missing manifest: {path}")
    return events, {"source": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                    "events": len(events), "ignored_incomplete_final_line": partial,
                    "complete": events[-1].get("type") == "end"}


def write_metrics(writer, events):
    writer.add_text("configuration", "```json\n" + json.dumps(events[0], indent=2) + "\n```", 0)
    writer.add_text("axes", "Accuracy uses measured training seconds (rounded) or optimizer updates. "
                    "Training diagnostics use rollout batches; group timings use group occurrences. "
                    "Export wall timestamps are not training timestamps. Disable smoothing to inspect measured values.", 0)
    for event in events[1:]:
        kind = event["type"]
        if kind == "eval":
            updates = event["optimizer_steps"]
            writer.add_scalar("eval/accuracy_by_training_seconds", event["accuracy"], round(event["training_s"]))
            writer.add_scalar("eval/accuracy_by_optimizer_updates", event["accuracy"], updates)
            writer.add_scalar("eval/measured_training_seconds", event["training_s"], updates)
            for key in ("truncation_fraction", "evaluation_s"):
                if key in event:
                    writer.add_scalar(f"eval/{key}", event[key], updates)
        elif kind == "update":
            batch = event["batches"]
            for key in ("mean_reward", "mixed_group_fraction", "grad_norm", "loss", "optimizer_stepped"):
                writer.add_scalar(f"train/{key}", event[f"batch_{key}"], batch)
            for key in ("optimizer_steps", "training_s", "generated_tokens", "discarded_tokens"):
                writer.add_scalar(f"train/{key}", event[key], batch)
            if event["generated_tokens"]:
                writer.add_scalar("train/wasted_token_fraction", event["discarded_tokens"] / event["generated_tokens"], batch)
            if event["groups"]:
                writer.add_scalar("train/replacement_fraction", event["replacements"] / event["groups"], batch)
            for key in ("batch_generation_s", "batch_update_s", "checkpoint_s", "budget_overrun_s"):
                if key in event:
                    writer.add_scalar(f"timing/{key}", event[key], batch)
        elif kind == "group":
            step = event["occurrence"]
            writer.add_scalar("timing/group_total_s", event["generation_s"], step)
            writer.add_scalar("timing/group_original_s", event["attempts"][0]["generation_s"], step)
            if event["replaced"]:
                writer.add_scalar("timing/group_retry_s", event["attempts"][1]["generation_s"], step)


def export(runs, output_dir):
    from torch.utils.tensorboard import SummaryWriter

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory for each snapshot; existing logs are preserved")
    snapshots = [(Path(run).expanduser().resolve().name, *read_events(run)) for run in runs]
    if not snapshots:
        raise ValueError("At least one run required")
    metadata = {"runs": [{"log_directory": f"{i:02d}-{name}", **source}
                          for i, (name, _, source) in enumerate(snapshots)]}
    # This guard validates the destination before SummaryWriter creates files.
    with external_output(output / "snapshot.json") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    for item, (_, events, _) in zip(metadata["runs"], snapshots):
        with SummaryWriter(log_dir=str(output / item["log_directory"])) as writer:
            write_metrics(writer, events)
    return {"output_dir": str(output), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory outside Git; repeat exports use new directories")
    args = parser.parse_args()
    print(json.dumps(export(args.runs, args.output_dir)))


if __name__ == "__main__":
    main()
