"""One on-policy, sequence-mean GRPO update, with no KL or replay epochs."""

import math
import time

import torch

from grpo import grpo_loss, response_logprobs
from replay import advantages


def update(model, optimizer, groups, device, *, max_grad_norm=1.0):
    """Average groups equally, then sequences equally within each group.

    Samples must come from the current policy. The detached same-forward
    denominator implements its first on-policy update. All-zero-advantage
    batches skip the optimizer (including momentum and weight decay); callers
    must count optimizer_stepped separately from rollout batches.
    """
    if not groups or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("Nonempty groups and a finite positive max_grad_norm required")
    device = torch.device(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Trainable parameters must be FP32")
    group_advantages = [advantages([s["reward"] for s in g["samples"]]) for g in groups]
    mixed = sum(any(a != 0 for a in adv) for adv in group_advantages)
    model.eval()  # Disable dropout without disabling autograd.
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_sum, grad_norm, backwards = 0.0, 0.0, 0
    try:
        for group, adv in zip(groups, group_advantages):
            scale = len(groups) * len(group["samples"])
            for sample, advantage in zip(group["samples"], adv):
                if advantage == 0:
                    continue
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logp = response_logprobs(model, group["prompt_token_ids"], sample["token_ids"])
                    loss = grpo_loss(logp, torch.ones_like(logp, dtype=torch.bool),
                                     logp.new_tensor([advantage])) / scale
                loss.backward()
                loss_sum += loss.detach().item()
                backwards += 1
        if mixed:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm,
                                                       error_if_nonfinite=True).item()
            optimizer.step()
    finally:
        optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = [s for g in groups for s in g["samples"]]
    return {
        "optimizer_stepped": bool(mixed), "loss": loss_sum, "grad_norm": grad_norm,
        "mixed_group_fraction": mixed / len(groups),
        "mean_reward": sum(sum(s["reward"] for s in g["samples"]) / len(g["samples"])
                           for g in groups) / len(groups),
        "groups": len(groups), "sequences": len(samples),
        "tokens": sum(len(s["token_ids"]) for s in samples),
        "backward_sequences": backwards, "update_s": time.perf_counter() - started,
    }
