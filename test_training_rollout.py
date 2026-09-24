"""CPU checks of real generate() and deterministic token-boundary controls."""

from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import training_rollout as rollout


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens):
        return " ".join(str(t) for t in tokens if t not in (0, 2))


class ScriptedModel(torch.nn.Module):
    def __init__(self, scripts):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.scripts = iter(scripts)
        self.calls = []

    def generate(self, input_ids, attention_mask, generation_config, stopping_criteria):
        self.calls.append((input_ids.tolist(), generation_config, torch.rand(1).item()))
        finished = torch.zeros(len(input_ids), dtype=torch.bool)
        for step in zip(*next(self.scripts)):
            tokens = torch.tensor(step)
            tokens[finished] = generation_config.pad_token_id
            input_ids = torch.cat((input_ids, tokens[:, None]), dim=1)
            finished |= tokens == generation_config.eos_token_id[0]
            stop = stopping_criteria(input_ids, None)
            if bool((finished | stop).all()) or input_ids.shape[1] - len(self.calls[-1][0][0]) >= generation_config.max_new_tokens:
                break
        return input_ids


def test_deadline_cancels_entire_group_and_retry_has_no_deadline(monkeypatch):
    times = iter([0, .1, .3, .31, 1, 1.1, 1.4, 1.41])
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(times))
    model = ScriptedModel([[[2, 8, 8], [3, 4, 2]], [[5, 2], [6, 2]]])
    result = rollout.retry_group(model, Tokenizer(), [1], group_size=2, max_tokens=4,
                                 seed=17, replacement_seed=18, deadline_s=.2)
    assert result["replaced"]
    assert result["discarded"]["samples"][0]["token_ids"] == [2]
    assert result["discarded"]["samples"][1]["token_ids"] == [3, 4]
    assert result["discarded_tokens"] == 3
    assert result["generated_tokens"] == 7
    assert result["padded_decode_tokens"] == 8
    assert result["discarded"]["overshoot_s"] == pytest.approx(.1)
    assert result["accepted"]["deadline_s"] is None
    assert not result["accepted"]["canceled"]
    assert result["generation_s"] == pytest.approx(.72)
    assert model.calls[0][0] == model.calls[1][0] == [[1], [1]]
    assert model.calls[0][2] != model.calls[1][2]


@pytest.mark.parametrize("tokens,cap,reason", [([[2], [2]], 4, "stop"), ([[3], [4]], 1, "length")])
def test_natural_completion_wins_over_deadline(monkeypatch, tokens, cap, reason):
    times = iter([0, 2, 2.1])
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(times))
    result = rollout.generate_group(ScriptedModel([tokens]), Tokenizer(), [1], group_size=2,
                                    max_tokens=cap, seed=2, deadline_s=1)
    assert not result["canceled"]
    assert result["stop_reason"] == reason


def test_eos_in_prompt_is_not_response_completion(monkeypatch):
    times = iter([0, 2, 2.1])
    monkeypatch.setattr(rollout.time, "perf_counter", lambda: next(times))
    result = rollout.generate_group(ScriptedModel([[[3, 2], [4, 2]]]), Tokenizer(), [2, 1],
                                    group_size=2, max_tokens=3, seed=2, deadline_s=1)
    assert result["canceled"]
    assert result["generated_tokens"] == 2


def test_random_replacement_finishes_original_and_discards_all_tokens():
    model = ScriptedModel([[[3, 2], [4, 2]], [[5, 2], [6, 2]]])
    result = rollout.retry_group(model, Tokenizer(), [1], group_size=2, max_tokens=4,
                                 seed=7, replacement_seed=8, random_replace=True)
    assert not result["discarded"]["canceled"]
    assert result["discarded_tokens"] == 4
    assert len(result["attempts"]) == 2


def test_real_transformers_generate_restores_rng_and_model_mode():
    torch.manual_seed(5)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=13, n_positions=16, n_embd=8,
                                      n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2))
    model.train()
    state = torch.get_rng_state().clone()
    kwargs = dict(group_size=3, max_tokens=5, seed=17)
    first = rollout.generate_group(model, Tokenizer(), [1, 4], **kwargs)
    assert torch.equal(torch.get_rng_state(), state)
    assert model.training
    second = rollout.generate_group(model, Tokenizer(), [1, 4], **kwargs)
    assert first["samples"] == second["samples"]
    assert all(1 <= sample["tokens"] <= 5 for sample in first["samples"])
    canceled = rollout.generate_group(model, Tokenizer(), [1, 4], deadline_s=0, **kwargs)
    assert canceled["canceled"]
    assert canceled["generated_tokens"] == 3
    greedy = rollout.generate_group(model, Tokenizer(), [1, 4], greedy=True, **kwargs)
    assert greedy["samples"][0] == greedy["samples"][1]
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("deadline", [-1, float("nan"), float("inf")])
def test_invalid_deadline(deadline):
    with pytest.raises(ValueError, match="Deadline"):
        rollout.generate_group(ScriptedModel([]), Tokenizer(), [1], group_size=2,
                               max_tokens=2, seed=1, deadline_s=deadline)
