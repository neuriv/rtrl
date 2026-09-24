import copy
from types import SimpleNamespace

import pytest
import torch

from grpo import grpo_loss, response_logprobs
from replay import advantages
from train_update import update


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(5, 5)
        self.dropout = torch.nn.Dropout(0.8)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.dropout(self.table(input_ids)))


def group(rewards, responses):
    return {"prompt_token_ids": [0, 1],
            "samples": [{"reward": r, "token_ids": ids} for r, ids in zip(rewards, responses)]}


def test_accumulation_matches_dense_equal_group_and_sequence_mean_update():
    torch.manual_seed(12)
    model = TinyPolicy()
    reference = copy.deepcopy(model).eval()
    groups = [group([0, 1], [[2], [3, 4, 2]]),
              group([0, 1, 1], [[1, 2], [4], [3, 2]]),
              group([1, 1], [[2], [4]])]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss = 0
    for g in groups:
        for sample, advantage in zip(g["samples"], advantages([s["reward"] for s in g["samples"]])):
            logp = response_logprobs(reference, g["prompt_token_ids"], sample["token_ids"])
            loss = loss + grpo_loss(logp, torch.ones_like(logp, dtype=torch.bool),
                                   logp.new_tensor([advantage])) / (len(groups) * len(g["samples"]))
    loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1000).item()
    reference_optimizer.step()
    before = model.table.weight.detach().clone()
    result = update(model, optimizer, groups, "cpu", max_grad_norm=1000)
    torch.testing.assert_close(model.table.weight, reference.table.weight)
    assert not torch.equal(before, model.table.weight)
    assert not model.training and not model.dropout.training
    assert all(p.grad is None for p in model.parameters())
    assert result["optimizer_stepped"] and result["backward_sequences"] == 5
    assert result["mixed_group_fraction"] == pytest.approx(2 / 3)
    assert result["mean_reward"] == pytest.approx((0.5 + 2 / 3 + 1) / 3)
    assert result["grad_norm"] == pytest.approx(expected_norm, rel=1e-6)
    assert result["loss"] == pytest.approx(loss.item(), abs=1e-7)
    assert result["groups"] == 3 and result["sequences"] == 7 and result["tokens"] == 11
    assert result["update_s"] > 0


def test_homogeneous_batch_skips_adamw_momentum_decay_and_clears_stale_gradients():
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    assert update(model, optimizer, [group([0, 1], [[2], [3]])], "cpu")["optimizer_stepped"]
    before = model.table.weight.detach().clone()
    step = optimizer.state[model.table.weight]["step"].item()
    model.table.weight.grad = torch.ones_like(model.table.weight)
    result = update(model, optimizer, [group([1, 1], [[2], [3]])], "cpu")
    assert not result["optimizer_stepped"]
    assert result["backward_sequences"] == result["grad_norm"] == result["loss"] == 0
    assert optimizer.state[model.table.weight]["step"].item() == step
    assert torch.equal(before, model.table.weight)
    assert model.table.weight.grad is None


def test_clipping_and_nonfinite_gradient_never_leave_stale_gradients():
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    groups = [group([0, 1], [[2], [3]])]
    before = model.table.weight.detach().clone()
    result = update(model, optimizer, groups, "cpu", max_grad_norm=0.01)
    assert result["grad_norm"] > 0.01
    assert (model.table.weight - before).norm().item() == pytest.approx(0.001, rel=1e-4)
    before = model.table.weight.detach().clone()
    hook = model.table.weight.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        update(model, optimizer, groups, "cpu")
    hook.remove()
    assert torch.equal(before, model.table.weight)
    assert model.table.weight.grad is None
