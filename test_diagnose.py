import copy

import pytest

from diagnose import diagnose


def bank(prompt, rewards, times):
    groups = [{"id": f"{prompt}:{trial}", "prompt_id": prompt, "trial": trial,
               "attempt": 0, "prompt_token_ids": [1, 2],
               "samples": [{"token_ids": [3], "logprobs": [-1.0], "reward": r,
                            "generation_s": t, "ready_s": t, "status": "complete"}
                           for r in rs]}
              for trial, (rs, t) in enumerate(zip(rewards, times))]
    manifest = {"model": "tiny", "revision": "abc", "sampling": {"temperature": 1.0},
                "group_size": 2, "backend": "transformers-mps", "dtype": "float32",
                "reward": "example:reward", "reward_source_sha256": "123"}
    return manifest, groups


def calibration():
    return bank("calibration", [[0, 1]] * 5, [1, 1, 1, 1, 1])


def test_zero_reward_shift_is_exact_null():
    result = diagnose(*bank("main", [[0, 1]] * 8, [1] * 4 + [2] * 4), *calibration())
    assert result["deadline_s"] == 1
    assert result["reward_shift"] == 0
    assert result["permutation"]["two_sided_tail"] == 1
    assert result["mixed_reward_group_shift"] == 0


def test_known_selection_signal_uses_whole_groups():
    manifest, groups = bank("main", [[0, 0]] * 12 + [[1, 1]] * 12, [1] * 12 + [2] * 12)
    groups[12]["samples"][0]["generation_s"] = 1
    result = diagnose(manifest, groups, *calibration())
    assert result["groups_admitted"] == 12
    assert result["reward_shift"] == -.5
    assert result["permutation"]["two_sided_tail"] < .005


def test_equal_prompt_weights_and_no_admission_are_explicit():
    manifest, first = bank("first", [[0, 0], [1, 1]], [1, 2])
    _, second = bank("second", [[0, 1]] * 8, [1] * 8)
    result = diagnose(manifest, first + second, *calibration())
    assert result["reward_shift"] == -.25  # Not group-count-weighted -.1.
    for g in second:
        for s in g["samples"]:
            s["generation_s"] = s["ready_s"] = 2
    result = diagnose(manifest, first + second, *calibration())
    assert result["prompts_without_admissions"] == ["second"]
    assert result["reward_shift"] is None
    assert result["permutation"] is None
    assert result["per_prompt"][1]["admitted_reward"] is None


@pytest.mark.parametrize("change,match", [("censored", "complete"), ("overlap", "disjoint"),
                                         ("retry", "attempt"), ("backend", "backend"),
                                         ("sampling", "sampling"), ("reward_source_sha256", "reward_source_sha256")])
def test_invalid_design_is_rejected(change, match):
    manifest, groups = bank("main", [[0, 1]], [1])
    cal_manifest, cal_groups = copy.deepcopy(calibration())
    if change == "censored":
        groups[0]["samples"][0].update(status="truncated", reward=None)
    elif change == "overlap":
        cal_groups[0]["prompt_id"] = "main"
    elif change == "retry":
        groups[0]["attempt"] = 1
    else:
        cal_manifest[change] = "different"
    with pytest.raises(ValueError, match=match):
        diagnose(manifest, groups, cal_manifest, cal_groups)
