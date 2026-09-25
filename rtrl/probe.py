"""CPU mechanism probes, not LLM/production evidence. Outputs stay outside Git."""

import math
import hashlib
from pathlib import Path

import numpy as np


def exact_binary(group_size, admission, policy=0.25):
    """Enumerate sufficient counts: fast-fail, fast-success, slow-fail, slow-success."""
    probabilities = np.array([(1-policy)*0.5, (1-policy)*0.5, policy*0.2, policy*0.8])
    rewards = np.array([0., 1., 0., 1.])
    score = np.array([0., 0., 1., 1.]) - policy
    full, accepted, mass = 0.0, 0.0, 0.0
    for c0 in range(group_size + 1):
        for c1 in range(group_size - c0 + 1):
            for c2 in range(group_size - c0 - c1 + 1):
                count = np.array([c0, c1, c2, group_size-c0-c1-c2])
                probability = math.exp(math.lgamma(group_size+1) - sum(math.lgamma(int(c)+1) for c in count)
                                       + float(count @ np.log(probabilities)))
                mean = float(count @ rewards / group_size)
                std = math.sqrt(group_size * mean * (1-mean) / (group_size-1))
                update = float((count * (rewards-mean) * score).sum() / (group_size * (std+1e-4)))
                kept = float(np.prod(np.asarray(admission)**count))
                full += probability * update
                accepted += probability * kept * update
                mass += probability * kept
    admitted = probabilities * np.asarray(admission)
    admitted /= admitted.sum()
    return {"group_size": group_size, "admission_by_outcome": list(admission),
            "admitted_slow_probability": float(admitted[2:].sum()),
            "admitted_reward_mean": float(admitted @ rewards),
            "complete_update": full, "admitted_update": accepted / mass if mass else None,
            "group_admission": mass, "unlimited_expected_attempts": 1/mass if mass else None}


def independent_delay_null(rng, replicates, group_admission, trials=64, actions=32, group_size=8):
    """Weak reward signal. Independent rejection has exactly zero population bias."""
    success = np.linspace(0.45, 0.55, actions)

    def sample_updates():
        choices = rng.integers(actions, size=(replicates, trials, group_size))
        rewards = (rng.random(choices.shape) < success[choices]).astype(float)
        adv = (rewards-rewards.mean(-1, keepdims=True)) / (rewards.std(-1, ddof=1, keepdims=True)+1e-4)
        gradient = np.zeros((replicates, trials, actions))
        r, t, _ = np.indices(choices.shape)
        np.add.at(gradient, (r, t, choices), adv / group_size)
        # The constant -1/actions score cancels because advantages sum to zero.
        return gradient

    first, fresh = sample_updates(), sample_updates()
    keep = rng.random((replicates, trials, 1)) < group_admission
    selected = np.where(keep, first, fresh)
    a, b = first.mean(1), selected.mean(1)
    difference = selected-first
    norm = np.linalg.norm(b-a, axis=-1)
    noise = np.sqrt(difference.var(1, ddof=1).sum(-1) / trials)
    cosine = (a*b).sum(-1) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    return {"group_admission": group_admission, "replicates": replicates, "trials_per_bank": trials,
            "actions": actions, "group_size": group_size, "success_probability_range": [0.45, 0.55],
            "population_bias": 0.0, "median_cosine": float(np.median(cosine)),
            "cosine_p05_p95": np.quantile(cosine, [.05, .95]).tolist(),
            "mean_paired_difference_norm": float(norm.mean()), "mean_estimated_rms_sampling_error": float(noise.mean()),
            "norm_of_replication_mean_difference": float((b-a).mean(0).dot((b-a).mean(0))**0.5)}


def retry_coverage(response_admission, group_size, attempts, trials):
    a = response_admission**group_size
    resolved = -math.expm1(attempts * math.log1p(-a)) if a < 1 else 1.0
    return {"response_admission": response_admission, "group_size": group_size,
            "attempts": attempts, "trials": trials, "group_admission": a,
            "trial_resolution": resolved, "all_trials_resolve": resolved**trials}


def run(args):
    if args.replicates < 2:
        raise ValueError("At least two replicates required")
    rng = np.random.default_rng(args.seed)
    # Match marginal per-response admission across mechanisms at 0.82.
    scenarios = {
        "independent": [0.82]*4,
        "strategy_only": [1., 1., 0.28, 0.28],
        "reward_only": [1., 1.-0.18/0.575, 1., 1.-0.18/0.575],
        "strategy_x_reward": [1., 1., 1., 0.1],
    }
    exact = [{"mechanism": name, **exact_binary(size, admit)}
             for size in [2, 4, 8, 16] for name, admit in scenarios.items()]
    # A separate ±1 group-update example isolates conditioning on resolved banks.
    caps = []
    for attempts in [1, 2, 4, 8, 16]:
        resolved = 1-0.5**attempts
        caps.append({"attempts": attempts, "unconditional_difference": 1.0,
                     "difference_after_discarding_unresolved_banks": 2-1/resolved})
    return {"kind": "CPU analytic/synthetic probes; no LLM or timing-performance claims",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "numpy": np.__version__,
            "seed": args.seed, "reward_std": "sample ddof=1, epsilon=1e-4",
            "binary_policy": {"slow_probability": .25, "fast_success": .5, "slow_success": .8, "equal_one_token_responses": True},
            "exact_binary": exact,
            "independent_delay_null": [independent_delay_null(rng, args.replicates, a) for a in [.1, .5, .9]],
            "finite_retry_coverage": [retry_coverage(.9, 8, a, 100) for a in [4, 8, 16, 32]],
            "discarded_bank_conditioning": caps}
