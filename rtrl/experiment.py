"""Run a versioned JSON recipe with data and outputs outside the checkout."""

import argparse
import hashlib
import json
from pathlib import Path

from .analyze_training import freeze_deadline, read_run
from .records import read_prompts
from .train import ATTENTION_BACKEND, parser as training_parser, run


def load_recipe(path, *, data_dir, output_dir, experiment_log, baseline=None, wandb_mode=None):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    recipe = json.loads(raw)
    if set(recipe) - {"arguments", "deadline_quantile", "reuse_initial_evaluation"}:
        raise ValueError("Unknown recipe fields")
    arguments = {key.replace("-", "_"): value for key, value in recipe["arguments"].items()}
    runtime = {"train_prompts", "eval_prompts", "output_dir", "experiment_log", "resume", "initial_eval"}
    if set(arguments) & runtime:
        raise ValueError("Data, output, log and checkpoint paths belong in runtime arguments")
    flags = []
    for key, value in arguments.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ValueError(f"Recipe argument {key} must be a string or number")
        flags.extend(["--" + key.replace("_", "-"), str(value)])
    data_dir = Path(data_dir).expanduser().resolve()
    flags.extend(["--train-prompts", str(data_dir / "train.jsonl"),
                  "--eval-prompts", str(data_dir / "eval.jsonl"),
                  "--output-dir", str(Path(output_dir).expanduser().resolve()),
                  "--experiment-log", str(Path(experiment_log).expanduser().resolve())])
    if wandb_mode is not None:
        flags.extend(["--wandb-mode", wandb_mode])
    args = training_parser().parse_args(flags)
    quantile = recipe.get("deadline_quantile")
    reuse = recipe.get("reuse_initial_evaluation", False)
    if not isinstance(reuse, bool):
        raise ValueError("reuse_initial_evaluation must be true or false")
    if quantile is not None or reuse:
        if baseline is None:
            raise ValueError("This recipe requires --baseline")
        reference = read_run(baseline)
        frozen = freeze_deadline(reference)
        if reference["config"].get("attention_backend") != ATTENTION_BACKEND:
            raise ValueError("Baseline configuration differs: attention_backend")
        for key in ("model", "revision", "seed", "group_size", "groups_per_update", "max_tokens", "learning_rate"):
            if getattr(args, key) != reference["config"][key]:
                raise ValueError(f"Baseline configuration differs: {key}")
        for key, data_path in (("train_sha256", args.train_prompts), ("eval_sha256", args.eval_prompts)):
            if read_prompts(data_path)[1] != reference["config"][key]:
                raise ValueError(f"Baseline data differs: {key}")
        if quantile is not None:
            if args.mode != "deadline" or args.deadline is not None or quantile not in frozen["deadlines_s"]:
                raise ValueError("Use one p50/p80/p90 calibration only for a deadline recipe")
            args.deadline = frozen["deadlines_s"][quantile]
        if reuse:
            args.initial_eval = str(Path(baseline).expanduser().resolve() / "events.jsonl")
    args.recipe_source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-log", required=True)
    parser.add_argument("--baseline", help="Completed matching baseline used for calibration or initial evaluation")
    parser.add_argument("--wandb-mode", choices=["online", "offline"])
    parser.add_argument("--print-config", action="store_true", help="Validate and print training arguments without running")
    options = vars(parser.parse_args())
    recipe, print_config = options.pop("recipe"), options.pop("print_config")
    args = load_recipe(recipe, **options)
    if print_config:
        print(json.dumps(vars(args), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
