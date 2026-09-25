"""Shared task, reward and retry-bank rules; generation stays in each backend."""

import hashlib
import importlib
import inspect
import math
import random
from pathlib import Path

from .records import read_run

PROTOCOL_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_reward(spec):
    module, function = spec.split(":")
    reward = getattr(importlib.import_module(module), function)
    source = inspect.getsourcefile(reward)
    if not source:
        raise ValueError("Verifier must have a Python source file for provenance")
    return reward, hashlib.sha256(Path(source).read_bytes()).hexdigest()


def tokenize_prompts(tokenizer, rows, max_tokens, max_model_len, raw=False):
    prompts = {}
    for row in rows:
        tokens = (tokenizer.encode(row["prompt"], add_special_tokens=True) if raw else
                  tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                tokenize=True, add_generation_prompt=True, return_dict=False))
        if not tokens or len(tokens) + max_tokens > max_model_len:
            raise ValueError(f"Prompt {row['id']} exceeds the explicit context budget; no silent truncation")
        prompts[row["id"]] = tokens
    return prompts


def score_sample(sample, row, reward, accept_length=False):
    sample.update(status="truncated", reward=None)
    if sample["finish_reason"] == "length" and not accept_length:
        return
    try:
        value = float(reward(row, sample["text"]))
        if not math.isfinite(value):
            raise ValueError("Verifier returned a nonfinite reward")
        sample.update(status="complete", reward=value)
    except Exception as error:
        sample.update(status="error", error=f"{type(error).__name__}: {error}")


def prepare_bank(args, rows, prompts):
    previous, old_groups = read_run(args.extend) if args.extend else (None, [])
    for group in old_groups:
        if prompts.get(group["prompt_id"]) != group["prompt_token_ids"]:
            raise ValueError("Extension changed the original prompt tokens")
    offset = previous["attempts"] if previous else 0
    jobs = [(row, trial, attempt) for row in rows for trial in range(args.trials)
            for attempt in range(offset, offset + args.attempts)]
    # Randomized dispatch must not change logical retry order.
    random.Random(args.seed).shuffle(jobs)
    return previous, old_groups, jobs


def check_extension(manifest, previous):
    if previous:
        mutable = {"attempts", "expected_groups", "trace_sha256", "extended_from"}
        changed = [key for key in manifest if key not in mutable and manifest[key] != previous.get(key)]
        if changed:
            raise ValueError(f"Extension must preserve the collection protocol: {', '.join(changed)}")
        manifest["extended_from"] = previous["trace_sha256"]
