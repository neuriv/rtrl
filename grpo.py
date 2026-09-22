"""First-update GRPO gradient diagnostics; no optimizer or parameter updates."""

import math
from itertools import groupby

import torch

from replay import advantages, complete, replay


def grpo_loss(logprobs, mask, advantage, old_logprobs=None, clip=0.2):
    """Sequence-mean GRPO, no KL. Same-scorer denominator isolates selection."""
    if logprobs.shape != mask.shape or mask.ndim != 2 or not mask.any(dim=1).all():
        raise ValueError("Each sequence needs a nonempty completion mask matching logprobs")
    old = logprobs.detach() if old_logprobs is None else old_logprobs.detach()
    # Mask before exponentiation: NaN padding multiplied by zero is still NaN.
    logprobs, old = logprobs.masked_fill(~mask, 0), old.masked_fill(~mask, 0)
    ratio = (logprobs - old).exp()
    advantage = advantage.detach()[:, None]
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    return -((objective * mask).sum(dim=1) / mask.sum(dim=1)).mean()


def response_logprobs(model, prompt, response):
    """Teacher-force original token IDs, including EOS; no decode/re-tokenize."""
    if not prompt or not response:
        raise ValueError("Nonempty prompt and response token IDs required")
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt + response], device=device, dtype=torch.long)
    logits = model(input_ids=tokens[:, :-1], use_cache=False).logits
    logits = logits[:, len(prompt)-1:, :].float()
    targets = tokens[:, len(prompt):]
    return logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)


def group_gradient(model, groups, parameters):
    """Average one group update per prompt/trial. Accumulate in FP32 on host."""
    accum = [torch.zeros(p.shape, dtype=torch.float32) for p in parameters]
    discrepancy_sum, discrepancy_max, token_count = 0.0, 0.0, 0
    for group in groups:
        adv = advantages([s["reward"] for s in group["samples"]])
        scale = len(groups) * len(group["samples"])
        for sample, a in zip(group["samples"], adv):
            logp = response_logprobs(model, group["prompt_token_ids"], sample["token_ids"])
            old = torch.tensor([sample["logprobs"]], device=logp.device)
            error = (logp.detach() - old).abs()
            discrepancy_sum += error.sum().item()
            discrepancy_max = max(discrepancy_max, error.max().item())
            token_count += error.numel()
            loss = grpo_loss(logp, torch.ones_like(logp, dtype=torch.bool), torch.tensor([a], device=logp.device))
            gradients = torch.autograd.grad(loss, parameters)
            for total, gradient in zip(accum, gradients):
                total.add_(gradient.detach().float().cpu(), alpha=1 / scale)
    return accum, {"mean_abs_logprob_error": discrepancy_sum / token_count,
                   "max_abs_logprob_error": discrepancy_max, "tokens": token_count}


def compare_gradients(baseline, selected):
    norm0, norm1, dot, delta = 0.0, 0.0, 0.0, 0.0
    # Convert before products. Chunking avoids a second full FP64 model on host.
    for a, b in zip(baseline, selected):
        for x, y in zip(a.reshape(-1).split(1 << 20), b.reshape(-1).split(1 << 20)):
            x, y = x.double(), y.double()
            norm0 += x.square().sum().item()
            norm1 += y.square().sum().item()
            dot += (x * y).sum().item()
            delta += (x - y).square().sum().item()
    norm0, norm1, delta = norm0**0.5, norm1**0.5, delta**0.5
    return {"baseline_norm": norm0, "selected_norm": norm1, "difference_norm": delta,
            "cosine": dot / (norm0 * norm1) if norm0 and norm1 else None,
            "relative_difference": delta / norm0 if norm0 else None}


def paired_summary(pairs):
    """Stream pairs sorted by prompt. Estimate trace of covariance within prompts."""
    total0 = total1 = None
    count, variance_sum, repeats = 0, 0.0, []
    for _, prompt_pairs in groupby(pairs, key=lambda row: row[0]):
        sum_delta, squared_delta, n = None, 0.0, 0
        for _, baseline, selected in prompt_pairs:
            if total0 is None:
                total0 = [torch.zeros_like(g) for g in baseline]
                total1 = [torch.zeros_like(g) for g in baseline]
            if sum_delta is None:
                sum_delta = [torch.zeros_like(g) for g in baseline]
            for t0, t1, delta, a, b in zip(total0, total1, sum_delta, baseline, selected):
                t0.add_(a)
                t1.add_(b)
                delta.add_(b-a)
            squared_delta += compare_gradients(baseline, selected)["difference_norm"]**2
            n += 1
        norm_sum = compare_gradients(sum_delta, sum_delta)["baseline_norm"]**2
        if n > 1:
            variance_sum += n * max(0.0, squared_delta - norm_sum/n) / (n-1)
        count += n
        repeats.append(n)
    if not count:
        raise ValueError("At least one paired trial required")
    result = compare_gradients([g/count for g in total0], [g/count for g in total1])
    rms = math.sqrt(variance_sum) / count if min(repeats) > 1 else None
    return {**result, "paired_trials": count, "minimum_trials_per_prompt": min(repeats),
            "rms_sampling_error": rms,
            "difference_over_rms_sampling_error": result["difference_norm"]/rms if rms else None,
            "uncertainty_assumption": "Independent trials within each prompt. RMS error is not a p-value; one trial per prompt cannot estimate it."}


def run(manifest, groups, args):
    result = replay(groups, args.deadline, args.clock)
    if result["unresolved"]:
        raise ValueError("Unresolved prompt/trials: use collect --extend on this trace. Dropping trials or restarting until all resolve biases the comparison")
    lookup = {g["id"]: g for g in groups}
    baseline = [lookup[d["baseline_id"]] for d in result["decisions"]]
    selected = [lookup[d["selected_id"]] for d in result["decisions"]]
    if not baseline or not all(complete(g) for g in baseline):
        raise ValueError("Full baseline is censored by failures/truncation; no valid paired audit")
    traversed = [lookup[gid] for d in result["decisions"] for gid in d["attempted_ids"]]
    if not all(complete(g) for g in traversed):
        raise ValueError("Retry history contains failure/truncation censoring; a timeout-only audit requires complete outcomes")
    from collect import cuda_device
    from transformers import AutoModelForCausalLM
    device = getattr(args, "device", "cuda")
    if device == "mps":
        if manifest.get("backend") != "transformers-mps" or manifest.get("dtype") != "float32":
            raise ValueError("MPS audits require an FP32 local trace")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Explicit MPS audit requested but MPS is unavailable")
        gpu, dtype = "Apple MPS (local feasibility)", torch.float32
    else:
        gpu, dtype = cuda_device(), getattr(torch, manifest["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=manifest["revision"], dtype=dtype,
        attn_implementation="sdpa",
    ).to(device).eval()
    head = model.get_output_embeddings().weight
    named = [(name, p) for name, p in model.named_parameters()
             if args.parameters == "all" or (args.parameters == "head" and p is head)
             or (args.parameters not in ("all", "head") and name.startswith(args.parameters))]
    if not named:
        raise ValueError("No parameters match the requested prefix")
    selected_ids = {id(p) for _, p in named}
    for p in model.parameters():
        p.requires_grad_(id(p) in selected_ids)
    params = [p for _, p in named]
    discrepancies = [{"sum": 0.0, "max_abs_logprob_error": 0.0, "tokens": 0} for _ in range(2)]

    def pairs():
        for b, s in sorted(zip(baseline, selected), key=lambda pair: pair[0]["prompt_id"]):
            g0, e0 = group_gradient(model, [b], params)
            g1, e1 = (g0, e0) if b["id"] == s["id"] else group_gradient(model, [s], params)
            for total, error in zip(discrepancies, [e0, e1]):
                total["sum"] += error["mean_abs_logprob_error"] * error["tokens"]
                total["tokens"] += error["tokens"]
                total["max_abs_logprob_error"] = max(total["max_abs_logprob_error"], error["max_abs_logprob_error"])
            yield b["prompt_id"], g0, g1

    gradient = paired_summary(pairs())
    for total in discrepancies:
        total["mean_abs_logprob_error"] = total.pop("sum") / total["tokens"]
    return {**result, "gradient": gradient, "gpu": gpu, "device": device, "dtype": str(dtype),
            "parameter_scope": args.parameters, "parameter_names": [n for n, _ in named],
            "parameter_count": sum(p.numel() for p in params),
            "tied_input_output_embeddings": model.get_input_embeddings().weight is head,
            "model": manifest["model"], "revision": manifest["revision"],
            "objective": "First update; sequence-mean GRPO; sample reward std + 1e-4; no KL; same-scorer detached denominator",
            "interpretation": "Gradient comparison in the named parameter block, not evidence of training quality or wall-clock speedup",
            "sampling_caveat": "A realized-bank gradient difference can be sampling noise. Do not discard incomplete banks and restart until all trials resolve.",
            "inference_vs_scorer": {"baseline": discrepancies[0], "selected": discrepancies[1]}}
