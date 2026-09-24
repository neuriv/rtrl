import json

import pytest

from analyze_training import MATCH_KEYS, compare, freeze_deadline, read_run


def run(mode="baseline", complete=True):
    config = {key: "same" for key in MATCH_KEYS}
    config.update(mode=mode, seconds=100, seed=17, code_commit="abc")
    curve = [{"optimizer_steps": step, "training_s": seconds, "accuracy": accuracy}
             for step, seconds, accuracy in [(0, 0, .25), (10, 80, .5), (20, 110, .75)]]
    return {"config": config, "groups": [{"attempts": [{"generation_s": t}]} for t in range(1, 15)],
            "summary": {"run": mode, "condition": mode, "seed": 17, "complete": complete,
                        "directory": "/external/run", "run_url": "https://wandb.ai/run", "curve": curve}}


def test_calibration_excludes_initial_groups_and_requires_completion():
    frozen = freeze_deadline(run())
    assert frozen["timing_sample_count"] == 10
    assert frozen["deadlines_s"] == {"p50": 9, "p80": 12, "p90": 13}
    with pytest.raises(ValueError, match="completed baseline"):
        freeze_deadline(run(complete=False))
    with pytest.raises(ValueError, match="completed baseline"):
        freeze_deadline(run("deadline"))


def test_only_measured_points_with_exact_common_steps():
    baseline, deadline = run(), run("deadline")
    deadline["summary"]["curve"][-1]["optimizer_steps"] = 30
    deadline["summary"]["curve"][1]["training_s"] = 60
    pair = compare([baseline, deadline], budget_seconds=90)["seed_pairs"][0]
    assert pair["exact_common_optimizer_step"] == 10
    assert [r["last_measured_at_or_before_budget"]["training_s"] for r in pair["runs"]] == [80, 60]
    assert all(r["at_common_update"]["optimizer_steps"] == 10 for r in pair["runs"])


def test_mismatched_data_is_not_compared():
    changed = run("deadline")
    changed["config"]["train_sha256"] = "other"
    with pytest.raises(ValueError, match="train_sha256"):
        compare([run(), changed])


def test_summary_counts_caps_only_in_accepted_samples(tmp_path):
    manifest = {"type": "manifest", **run()["config"], "run_url": "https://wandb.ai/run"}
    accepted = {"generation_s": 3, "samples": [{"reward": 0, "finish_reason": "stop"},
                                               {"reward": 1, "finish_reason": "stop"}]}
    discarded = {"generation_s": 2, "samples": [{"reward": 0, "finish_reason": "length"}]}
    group = {"type": "group", "accepted": accepted, "attempts": [discarded, accepted],
             "generated_tokens": 20, "discarded_tokens": 5, "replaced": True, "generation_s": 5}
    end = {"type": "end", "training_s": 102, "optimizer_steps": 10}
    (tmp_path / "events.jsonl").write_text("\n".join(map(json.dumps, [manifest, group, end])))
    summary = read_run(tmp_path)["summary"]
    assert summary["complete"] and summary["replacement_rate"] == 1
    assert summary["accepted_cap_fraction"] == 0 and summary["mixed_group_fraction"] == 1
    assert summary["wasted_token_fraction"] == .25 and summary["budget_overrun_s"] == 2
    assert summary["mean_original_attempt_s"] == 2 and summary["mean_retry_s"] == 3
