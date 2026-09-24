"""Measured training comparisons and frozen empirical baseline deadlines."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from records import external_output

MATCH_KEYS = ("model", "revision", "train_sha256", "eval_sha256", "group_size",
              "groups_per_update", "max_tokens", "learning_rate", "precision",
              "objective", "cache", "scheduler", "cap_reward", "sampling", "reward")


def read_run(directory):
    directory = Path(directory).expanduser().resolve()
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines() if line]
    if not events or events[0].get("type") != "manifest":
        raise ValueError(f"Missing manifest: {directory}")
    config = events[0]
    groups = [event for event in events if event["type"] == "group"]
    evaluations = [{key: event[key] for key in ("optimizer_steps", "training_s", "accuracy",
                                               "truncation_fraction")}
                   for event in events if event["type"] == "eval"]
    states = [event for event in events if event["type"] in ("update", "eval", "end")]
    state = states[-1] if states else {}
    samples = [sample for group in groups for sample in group["accepted"]["samples"]]
    generated = sum(group["generated_tokens"] for group in groups)
    retries = [group["attempts"][1]["generation_s"] for group in groups if group["replaced"]]
    summary = {
        "run": directory.name, "directory": str(directory), "condition": config["mode"],
        "seed": config["seed"], "deadline_s": config.get("deadline"),
        "requested_replacement_rate": config.get("replacement_rate"),
        "run_url": config["run_url"], "code_commit": config["code_commit"],
        "complete": events[-1]["type"] == "end", "curve": evaluations,
        "training_s": state.get("training_s", 0), "budget_s": config["seconds"],
        "optimizer_steps": state.get("optimizer_steps", 0), "accepted_groups": len(groups),
        "budget_overrun_s": max(0, state.get("training_s", 0) - config["seconds"]),
        "replacement_rate": sum(g["replaced"] for g in groups) / len(groups) if groups else None,
        "wasted_token_fraction": sum(g["discarded_tokens"] for g in groups) / generated if generated else None,
        "generated_tokens": generated, "discarded_tokens": sum(g["discarded_tokens"] for g in groups),
        "mixed_group_fraction": mean(len({s["reward"] for s in g["accepted"]["samples"]}) > 1
                                     for g in groups) if groups else None,
        "accepted_cap_fraction": mean(s["finish_reason"] == "length" for s in samples) if samples else None,
        "mean_original_attempt_s": mean(g["attempts"][0]["generation_s"] for g in groups) if groups else None,
        "mean_retry_s": mean(retries) if retries else None,
        "mean_group_total_s": mean(g["generation_s"] for g in groups) if groups else None,
        "evaluation_s": state.get("evaluation_s"), "checkpoint_s": state.get("checkpoint_s"),
        "wall_s": state.get("wall_s"),
    }
    return {"config": config, "groups": groups, "summary": summary}


def freeze_deadline(run):
    config, summary = run["config"], run["summary"]
    if config["mode"] != "baseline" or not summary["complete"]:
        raise ValueError("Deadline calibration requires a completed baseline with an end marker")
    timings = sorted(group["attempts"][0]["generation_s"] for group in run["groups"][4:])
    if not timings or any(not math.isfinite(t) or t <= 0 for t in timings):
        raise ValueError("Calibration needs positive original timings after the first 4 groups")
    return {"baseline_config": config, "baseline_directory": summary["directory"],
            "run_url": summary["run_url"], "code_commit": config["code_commit"], "seed": config["seed"],
            "excluded_initial_groups": 4, "timing_sample_count": len(timings),
            "estimator": "Empirical nearest-rank quantile of original generation_s",
            "primary_quantile": "p80", "deadlines_s": {
                f"p{percent}": timings[math.ceil(len(timings) * percent / 100) - 1]
                for percent in (50, 80, 90)}}


def compare(runs, budget_seconds=None):
    first = runs[0]["config"]
    for run in runs[1:]:
        mismatch = [key for key in MATCH_KEYS if run["config"][key] != first[key]]
        if mismatch:
            raise ValueError(f"Incompatible experiment configurations: {', '.join(mismatch)}")
    budget = min(run["config"]["seconds"] for run in runs)
    if budget_seconds is not None:
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("budget-seconds must be finite and positive")
        budget = min(budget, budget_seconds)
    summaries = [run["summary"] for run in runs]
    paired = []
    for seed in sorted({run["seed"] for run in summaries}):
        members = [run for run in summaries if run["seed"] == seed]
        common = set.intersection(*[{p["optimizer_steps"] for p in run["curve"]} for run in members])
        step = max(common) if common else None
        points = []
        for run in members:
            within_budget = [p for p in run["curve"] if p["training_s"] <= budget]
            exact = [p for p in run["curve"] if p["optimizer_steps"] == step]
            points.append({"run": run["run"], "condition": run["condition"],
                           "complete": run["complete"], "run_url": run["run_url"],
                           "at_common_update": exact[-1] if exact else None,
                           "last_measured_at_or_before_budget": within_budget[-1] if within_budget else None})
        paired.append({"seed_pair_label": f"seed-{seed}", "exact_common_optimizer_step": step,
                       "budget_s": budget, "runs": points})
    return {"matched_configuration": {key: first[key] for key in MATCH_KEYS},
            "runs": summaries, "seed_pairs": paired,
            "interpretation": [
                "Accuracy points are measured; no time or update interpolation is used.",
                "Budget comparisons use each run's last evaluation at or below the common budget; observed times may differ.",
                "Training time includes original generation, discarded work, retries and optimization; evaluation/checkpoint time is separate.",
                "Random replacement is independent of completion time and finishes discarded originals; interpret selection effects at equal updates.",
                "Individual paired seeds are descriptive; these summaries establish no statistical significance."]}


def save(path, value):
    with external_output(path) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-deadline")
    parser.add_argument("--budget-seconds", type=float)
    args = parser.parse_args()
    runs = [read_run(directory) for directory in args.runs]
    report = compare(runs, args.budget_seconds)
    if args.freeze_deadline:
        baselines = [run for run in runs if run["config"]["mode"] == "baseline"]
        if len(baselines) != 1:
            raise ValueError("Specify exactly one baseline when freezing a deadline")
        save(args.freeze_deadline, freeze_deadline(baselines[0]))
    save(args.output, report)
    print(json.dumps({"output": args.output, "runs": len(runs), "freeze_deadline": args.freeze_deadline}))


if __name__ == "__main__":
    main()
