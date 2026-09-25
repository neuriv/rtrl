import hashlib
import json

import pytest

from rtrl.experiment import load_recipe


@pytest.fixture
def recipe_inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    hashes = {}
    for split in ("train", "eval"):
        path = data / f"{split}.jsonl"
        path.write_text(json.dumps({"id": split, "prompt": "40 + 2", "reference": "42"}) + "\n")
        hashes[split + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = {"model": "tiny", "revision": "a" * 40, "seed": 17, "group_size": 4,
                 "groups_per_update": 4, "max_tokens": 512, "learning_rate": 1e-6}
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    manifest = {"type": "manifest", **arguments, **hashes, "mode": "baseline", "seconds": 1800,
                "attention_backend": {"implementation": "sdpa", "cudnn": False,
                                      "flash": True, "efficient": True, "math": True},
                "run_url": None, "code_commit": "baseline-commit"}
    groups = [{"type": "group", "accepted": {"samples": [{"reward": 0, "finish_reason": "stop"}]},
               "generated_tokens": 1, "discarded_tokens": 0, "replaced": False,
               "generation_s": t, "attempts": [{"generation_s": t}]}
              for t in (100, 100, 100, 100, 1, 2, 3, 4, 5)]
    records = [manifest, *groups, {"type": "end", "training_s": 30, "optimizer_steps": 2}]
    (baseline / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    recipe = tmp_path / "recipe.json"
    options = dict(data_dir=data, output_dir=tmp_path / "run", experiment_log=tmp_path / "log.md")
    return recipe, arguments, baseline, options


def test_recipe_preserves_arguments_and_records_source(recipe_inputs):
    path, arguments, baseline, options = recipe_inputs
    path.write_text(json.dumps({"arguments": {**arguments, "mode": "baseline", "seconds": 1800}}))
    args = load_recipe(path, **options, wandb_mode="offline")
    assert args.mode == "baseline" and args.seconds == 1800
    assert args.train_prompts == str(options["data_dir"] / "train.jsonl")
    assert args.wandb_mode == "offline"
    assert args.recipe_source == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_calibration_uses_baseline_rule_and_can_reuse_initial_evaluation(recipe_inputs):
    path, arguments, baseline, options = recipe_inputs
    path.write_text(json.dumps({"arguments": {**arguments, "mode": "deadline"},
                                "deadline_quantile": "p80", "reuse_initial_evaluation": True}))
    args = load_recipe(path, **options, baseline=baseline)
    assert args.deadline == 4  # Existing rule excludes the first four groups.
    assert args.initial_eval == str(baseline / "events.jsonl")
    assert args.recipe_source == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="requires --baseline"):
        load_recipe(path, **options)
    (options["data_dir"] / "eval.jsonl").write_text(
        json.dumps({"id": "eval", "prompt": "different", "reference": "42"}) + "\n")
    with pytest.raises(ValueError, match="Baseline data differs: eval_sha256"):
        load_recipe(path, **options, baseline=baseline)


def test_calibration_rejects_changed_model_or_incomplete_baseline(recipe_inputs):
    path, arguments, baseline, options = recipe_inputs
    recipe = {"arguments": {**arguments, "mode": "deadline", "revision": "b" * 40}, "deadline_quantile": "p50"}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match="Baseline configuration differs: revision"):
        load_recipe(path, **options, baseline=baseline)
    recipe["arguments"]["revision"] = arguments["revision"]
    path.write_text(json.dumps(recipe))
    events = baseline / "events.jsonl"
    events.write_text("\n".join(events.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="completed baseline"):
        load_recipe(path, **options, baseline=baseline)


@pytest.mark.parametrize("backend", [None, {"implementation": "sdpa", "cudnn": True,
                                         "flash": True, "efficient": True, "math": True}])
def test_calibration_rejects_unknown_or_superseded_attention_backend(recipe_inputs, backend):
    path, arguments, baseline, options = recipe_inputs
    path.write_text(json.dumps({"arguments": {**arguments, "mode": "deadline"}, "deadline_quantile": "p80"}))
    events = baseline / "events.jsonl"
    records = [json.loads(line) for line in events.read_text().splitlines()]
    records[0]["attention_backend"] = backend
    events.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="Baseline configuration differs: attention_backend"):
        load_recipe(path, **options, baseline=baseline)


@pytest.mark.parametrize("argument", ["output_dir", "train_prompts", "initial_eval", "initial-eval", "resume"])
def test_recipe_cannot_hide_runtime_paths(recipe_inputs, argument):
    path, arguments, baseline, options = recipe_inputs
    path.write_text(json.dumps({"arguments": {**arguments, argument: "hidden"}}))
    with pytest.raises(ValueError, match="runtime arguments"):
        load_recipe(path, **options)
