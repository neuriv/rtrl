"""Exploratory admission diagnostics on complete, unconditional frozen-policy groups."""

import math
import random
from statistics import fmean

from records import validate_groups
from replay import complete


def diagnose(manifest, groups, calibration_manifest, calibration_groups, *, permutations=2000, seed=17):
    """Fix a deadline on separate prompts; compare admitted vs all within each prompt."""
    if not groups or not calibration_groups or permutations < 1:
        raise ValueError("Nonempty main/calibration groups and positive permutations required")
    if manifest.get("backend") != "transformers-mps" or manifest["backend"] != calibration_manifest.get("backend"):
        raise ValueError("An explicit matching backend is required for compatible timing")
    for field in ("model", "revision", "sampling", "group_size", "dtype", "reward", "reward_source_sha256"):
        if manifest[field] != calibration_manifest[field]:
            raise ValueError(f"Calibration changed {field}")
    for bank in (groups, calibration_groups):
        if any(g["attempt"] != 0 for g in bank):
            raise ValueError("Only unconditional attempt0 groups belong in this diagnostic")
        if not all(complete(g) for g in bank):
            raise ValueError("All outcomes must be complete; no censoring or failure filtering")
        validate_groups(bank)
        if any(len(g["samples"]) != manifest["group_size"] for g in bank):
            raise ValueError("Group size differs from the manifest")
    prompt_ids = {g["prompt_id"] for g in groups}
    if prompt_ids & {g["prompt_id"] for g in calibration_groups}:
        raise ValueError("Calibration and main prompt IDs must be disjoint")
    times = sorted(max(s["generation_s"] for s in g["samples"]) for g in calibration_groups)
    rank = .8 * (len(times) - 1)
    lower = math.floor(rank)
    deadline = times[lower] + (rank - lower) * (times[min(lower + 1, len(times) - 1)] - times[lower])
    buckets = {p: [] for p in sorted(prompt_ids)}
    for group in groups:
        rewards = [s["reward"] for s in group["samples"]]
        buckets[group["prompt_id"]].append((fmean(rewards), float(min(rewards) != max(rewards)),
                                           max(s["generation_s"] for s in group["samples"]) <= deadline))
    per_prompt, permutation_rows = [], []
    for prompt, rows in buckets.items():
        admitted = [row for row in rows if row[2]]
        all_reward, all_mixed = fmean(r[0] for r in rows), fmean(r[1] for r in rows)
        admitted_reward = fmean(r[0] for r in admitted) if admitted else None
        admitted_mixed = fmean(r[1] for r in admitted) if admitted else None
        per_prompt.append({"prompt_id": prompt, "groups": len(rows), "admitted": len(admitted),
                           "all_reward": all_reward, "admitted_reward": admitted_reward,
                           "reward_shift": admitted_reward - all_reward if admitted else None,
                           "all_mixed_reward_fraction": all_mixed,
                           "admitted_mixed_reward_fraction": admitted_mixed,
                           "mixed_reward_group_shift": admitted_mixed - all_mixed if admitted else None})
        permutation_rows.append(([r[0] for r in rows], len(admitted), all_reward))
    missing = [p["prompt_id"] for p in per_prompt if not p["admitted"]]
    shift = fmean(p["reward_shift"] for p in per_prompt) if not missing else None
    mixed_shift = fmean(p["mixed_reward_group_shift"] for p in per_prompt) if not missing else None
    permutation = None
    if not missing:
        rng = random.Random(seed)
        null = [fmean(fmean(rng.sample(rewards, count)) - baseline
                      for rewards, count, baseline in permutation_rows) for _ in range(permutations)]
        permutation = {"replicates": permutations, "seed": seed,
                       "two_sided_tail": (1 + sum(abs(x) >= abs(shift) - 1e-12 for x in null)) / (permutations + 1),
                       "null_mean": fmean(null),
                       "method": "Within-prompt random admission masks, preserving observed admitted counts; plus-one Monte Carlo tail"}
    return {"scope": "Conditional admission association only; no retries, GRPO gradients, learning effect, or throughput claim",
            "deadline_s": deadline, "deadline_rule": "Calibration 80th percentile of group-max generation_s; linear interpolation; boundary admitted",
            "calibration_groups": len(calibration_groups), "calibration_prompts": len({g["prompt_id"] for g in calibration_groups}),
            "groups": len(groups), "groups_admitted": sum(p["admitted"] for p in per_prompt),
            "prompts": len(per_prompt), "prompts_without_admissions": missing,
            "weighting": "Equal main-prompt weights; aggregate undefined if any prompt has no admissions",
            "reward_shift": shift, "mixed_reward_group_shift": mixed_shift,
            "per_prompt": per_prompt, "permutation": permutation,
            "uncertainty": "Exploratory conditional permutation diagnostic, not an effect-size confidence interval. Exchangeability within prompts is required; shared load or thermal drift can invalidate it. Fixed prompts do not establish generalization. Calibration quantile uncertainty is not included."}
