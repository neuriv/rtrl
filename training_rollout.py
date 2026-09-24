"""Live, serial groups of batched siblings; one model shared with the trainer."""

import math
import time

import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GroupDeadline(StoppingCriteria):
    """Cancel at a token boundary, after checking whole-group natural completion."""

    def __init__(self, prompt_length, max_tokens, eos_ids, started, deadline_s):
        self.prompt_length, self.max_tokens = prompt_length, max_tokens
        self.eos_ids, self.started, self.deadline_s = eos_ids, started, deadline_s
        self.finished = None
        self.canceled = False
        self.first_token_s = None
        self.stop_elapsed_s = None

    def __call__(self, input_ids, scores, **kwargs):
        synchronize(input_ids.device)
        elapsed = time.perf_counter() - self.started
        if self.first_token_s is None:
            self.first_token_s = elapsed
        if self.finished is None:
            self.finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for eos in self.eos_ids:
            self.finished |= input_ids[:, -1] == eos
        natural = input_ids.shape[1] - self.prompt_length >= self.max_tokens or bool(self.finished.all())
        if natural or (self.deadline_s is not None and elapsed > self.deadline_s):
            self.stop_elapsed_s = elapsed
            self.canceled = not natural
        return torch.full_like(self.finished, self.canceled)


def generate_group(model, tokenizer, prompt_ids, *, group_size, max_tokens, seed,
                   deadline_s=None, greedy=False):
    """Generate a group. Caller controls precision (e.g. BF16 autocast).

    Tokens include the first EOS and exclude later padding. A canceled group may
    contain finished siblings: every token in that group is still discarded.
    Timing includes generation setup/prefill/decode, but excludes text decoding.
    Finished batch rows still consume compute, reported as padded_decode_tokens.
    """
    if not prompt_ids or group_size < 1 or max_tokens < 1:
        raise ValueError("Nonempty prompt, positive group size and token budget required")
    if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s < 0):
        raise ValueError("Deadline must be finite and nonnegative")
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos_ids[0] if eos_ids else 0
    config = GenerationConfig(
        max_new_tokens=max_tokens, do_sample=not greedy, temperature=1.0,
        top_p=1.0, top_k=0, repetition_penalty=1.0,
        eos_token_id=eos_ids or None, pad_token_id=pad,
        use_cache=True, disable_compile=True,
    )
    inputs = torch.tensor([prompt_ids] * group_size, device=device, dtype=torch.long)
    attention = torch.ones_like(inputs)
    training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            synchronize(device)
            started = time.perf_counter()
            deadline = GroupDeadline(len(prompt_ids), max_tokens, eos_ids, started, deadline_s)
            sequences = model.generate(
                input_ids=inputs, attention_mask=attention, generation_config=config,
                stopping_criteria=StoppingCriteriaList([deadline]),
            )
            synchronize(device)
            elapsed = time.perf_counter() - started
    finally:
        model.train(training)
    responses = sequences[:, len(prompt_ids):].tolist()
    samples = []
    for tokens in responses:
        eos_position = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        if eos_position is not None:
            tokens = tokens[:eos_position + 1]
        samples.append({"token_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
                        "tokens": len(tokens), "finish_reason": "stop" if eos_position is not None else
                        ("deadline" if deadline.canceled else "length")})
    return {"samples": samples, "seed": seed, "generation_s": elapsed,
            "generated_tokens": sum(sample["tokens"] for sample in samples),
            "padded_decode_tokens": group_size * (sequences.shape[1] - len(prompt_ids)),
            "first_token_s": deadline.first_token_s, "canceled": deadline.canceled,
            "deadline_s": deadline_s,
            "overshoot_s": max(0.0, deadline.stop_elapsed_s - deadline_s) if deadline.canceled else 0.0,
            "stop_reason": "deadline" if deadline.canceled else
            ("stop" if all(sample["finish_reason"] == "stop" for sample in samples) else "length")}


def retry_group(model, tokenizer, prompt_ids, *, group_size, max_tokens, seed,
                replacement_seed, deadline_s=None, random_replace=False):
    """At most one retry. A caller-supplied random decision is time-independent."""
    if seed == replacement_seed:
        raise ValueError("Replacement must use a fresh seed")
    first = generate_group(model, tokenizer, prompt_ids, group_size=group_size,
                           max_tokens=max_tokens, seed=seed, deadline_s=deadline_s)
    discarded = first if first["canceled"] or random_replace else None
    accepted = generate_group(model, tokenizer, prompt_ids, group_size=group_size,
                              max_tokens=max_tokens, seed=replacement_seed) if discarded else first
    attempts = [first, accepted] if discarded else [first]
    return {"accepted": accepted, "discarded": discarded, "attempts": attempts,
            "replaced": discarded is not None,
            "generation_s": sum(attempt["generation_s"] for attempt in attempts),
            "generated_tokens": sum(attempt["generated_tokens"] for attempt in attempts),
            "padded_decode_tokens": sum(attempt["padded_decode_tokens"] for attempt in attempts),
            "discarded_tokens": discarded["generated_tokens"] if discarded else 0}
