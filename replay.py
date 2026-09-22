"""Selection replay on completed traces; it does not simulate a live scheduler."""

import math
from statistics import mean, stdev

from records import validate_groups


def advantages(rewards):
    """Sample standard deviation (ddof=1), epsilon=1e-4; equal rewards give zero."""
    if len(rewards) < 2 or not all(math.isfinite(r) for r in rewards):
        raise ValueError("At least two finite rewards required")
    center, scale = mean(rewards), stdev(rewards) + 1e-4
    return [(r - center) / scale for r in rewards]


def complete(group):
    return all(s["status"] == "complete" for s in group["samples"])


def replay(groups, deadline=None, clock="ready_s"):
    if deadline is not None and (not math.isfinite(deadline) or deadline < 0):
        raise ValueError("Deadline must be finite and nonnegative, or omitted")
    if clock not in ("ready_s", "generation_s"):
        raise ValueError("Clock must be ready_s or generation_s")
    trials = validate_groups(groups)
    decisions = []
    for (prompt_id, trial), attempts in sorted(trials.items()):
        attempts = sorted(attempts, key=lambda g: g["attempt"])
        tried, selected = [], None
        for group in attempts:
            tried.append(group["id"])
            if complete(group) and (deadline is None or max(s[clock] for s in group["samples"]) <= deadline):
                selected = group
                break
        decisions.append({
            "prompt_id": prompt_id, "trial": trial, "baseline_id": attempts[0]["id"],
            "selected_id": selected["id"] if selected else None, "attempted_ids": tried,
            "advantages": advantages([s["reward"] for s in selected["samples"]]) if selected else None,
        })
    attempted = sum(len(d["attempted_ids"]) for d in decisions)
    resolved = sum(d["selected_id"] is not None for d in decisions)
    return {
        "scope": "Offline selection only. Timings include uncancelled work; no live throughput estimate.",
        "deadline_s": deadline, "clock": clock, "trials": len(decisions),
        "resolved": resolved, "unresolved": len(decisions) - resolved,
        "attempted_groups": attempted, "rejected_groups": attempted - resolved,
        "decisions": decisions,
    }
