import math
import asyncio
import itertools
import json
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from replay import advantages, replay
from grpo import grpo_loss, response_logprobs, group_gradient, compare_gradients
from records import external_output, read_run
from collect import collect_group


def group(prompt="p", attempt=0, times=(1.0, 2.0), rewards=(0.0, 1.0)):
    return {
        "id": f"{prompt}:0:{attempt}", "prompt_id": prompt, "trial": 0,
        "attempt": attempt, "prompt_token_ids": [1, 2],
        "samples": [{"token_ids": [3], "logprobs": [-1.0], "reward": r,
                     "generation_s": t, "ready_s": t, "status": "complete",
                     "finish_reason": "stop"} for t, r in zip(times, rewards)],
    }


def test_whole_group_retry_boundary_and_exhaustion():
    first = group(times=(1, 3))
    second = group(attempt=1, times=(2, 2))
    result = replay([second, first], deadline=2)
    assert result["decisions"][0]["selected_id"] == second["id"]
    assert result["decisions"][0]["attempted_ids"] == [first["id"], second["id"]]
    assert replay([first, second], deadline=1)["unresolved"] == 1
    assert replay([first, second], deadline=None)["decisions"][0]["selected_id"] == first["id"]


def test_group_normalization_and_zero_variance():
    assert advantages([1, 1, 1]) == [0, 0, 0]
    expected = 0.5 / (math.sqrt(0.5) + 1e-4)
    assert advantages([0, 1]) == pytest.approx([-expected, expected])


def test_grpo_masks_prompt_padding_and_detaches_behavior():
    logp = torch.tensor([[-3., -2., -1.], [-4., -3., -2.]], requires_grad=True)
    old = logp.detach().clone().requires_grad_()
    adv = torch.tensor([1., -1.], requires_grad=True)
    mask = torch.tensor([[False, True, True], [False, True, False]])
    grpo_loss(logp, mask, adv, old).backward()
    assert logp.grad.tolist() == [[0, -0.25, -0.25], [0, 0.5, 0]]
    assert old.grad is None and adv.grad is None


def test_missing_attempts_and_changed_prompt_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        replay([group(), group(attempt=2)])
    changed = group(attempt=1)
    changed["prompt_token_ids"] = [9]
    with pytest.raises(ValueError, match="preserve"):
        replay([group(), changed])


def test_unresolved_prompt_does_not_disappear():
    result = replay([group("fast"), group("slow", times=(4, 5))], deadline=2)
    assert (result["trials"], result["resolved"], result["unresolved"]) == (2, 1, 1)
    assert result["decisions"][1]["selected_id"] is None


def test_missing_reward_is_not_task_failure_and_clock_is_explicit():
    bad = group()
    bad["samples"][0].update(status="error", reward=None)
    assert replay([bad], deadline=None)["unresolved"] == 1
    good = group()
    good["samples"][0]["ready_s"] = 4
    assert replay([good], 2, "ready_s")["unresolved"] == 1
    assert replay([good], 2, "generation_s")["resolved"] == 1


@pytest.mark.parametrize("deadline", [-1, float("nan"), float("inf")])
def test_invalid_deadline(deadline):
    with pytest.raises(ValueError):
        replay([group()], deadline)


def test_independent_delays_preserve_expected_score_update():
    # Enumerate all actions and independent delays, rather than sampling noise.
    full, admitted = [], []
    for actions in itertools.product([0., 1.], repeat=2):
        update = sum(a * (action - 0.5) for a, action in zip(advantages(actions), actions)) / 2
        for delays in itertools.product([1., 3.], repeat=2):
            full.append(update)
            if replay([group(times=delays, rewards=actions)], deadline=2)["resolved"]:
                admitted.append(update)
    assert sum(full) / len(full) == pytest.approx(sum(admitted) / len(admitted))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.1, 0.3, -0.2], [-0.1, 0.2, 0.4], [0.5, 0.1, -0.3]]))

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.weight[input_ids])


def test_teacher_forcing_excludes_prompt_and_includes_final_token():
    model = TinyLM()
    actual = response_logprobs(model, [0, 1], [2, 0])
    expected = model.weight.log_softmax(-1)[[1, 2], [2, 0]]
    torch.testing.assert_close(actual[0], expected)


def test_grpo_gradient_matches_finite_difference():
    model = TinyLM()
    old = response_logprobs(model, [0], [1, 2]).detach()
    mask = torch.tensor([[True, True]])

    def loss():
        return grpo_loss(response_logprobs(model, [0], [1, 2]), mask, torch.tensor([0.7]), old)

    loss().backward()
    analytical = model.weight.grad[0, 1].item()
    epsilon = 1e-3
    with torch.no_grad():
        original = model.weight[0, 1].item()
        model.weight[0, 1] = original + epsilon
        plus = loss().item()
        model.weight[0, 1] = original - epsilon
        minus = loss().item()
        model.weight[0, 1] = original
    assert analytical == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-5)


def test_clipping_and_zero_advantage():
    logp = torch.tensor([[0.0]], requires_grad=True)
    old = torch.tensor([[-1.0]])
    grpo_loss(logp, torch.tensor([[True]]), torch.tensor([1.0]), old).backward()
    assert logp.grad.item() == 0
    logp.grad = None
    grpo_loss(logp, torch.tensor([[True]]), torch.tensor([0.0])).backward()
    assert logp.grad.item() == 0


def test_paired_audit_identity_and_zero_signal():
    model = TinyLM()
    g = group()
    g["prompt_token_ids"] = [0]
    for s, token in zip(g["samples"], [1, 2]):
        s["token_ids"] = [token]
        s["logprobs"] = response_logprobs(model, [0], [token]).detach()[0].tolist()
    gradient, error = group_gradient(model, [g], [model.weight])
    assert compare_gradients(gradient, gradient)["difference_norm"] == 0
    assert error["max_abs_logprob_error"] == 0
    for s in g["samples"]:
        s["reward"] = 1.0
    zero, _ = group_gradient(model, [g], [model.weight])
    assert compare_gradients(zero, zero)["cosine"] is None
    assert zero[0].count_nonzero() == 0


class FakeBackend:
    def __init__(self, finish="stop"):
        self.calls = []
        self.finish = finish

    async def generate(self, prompt, seed, request_id, started, max_tokens=None):
        self.calls.append((list(prompt), seed, request_id))
        return {"token_ids": [1], "logprobs": [-0.5], "text": "answer",
                "first_token_s": 0.0, "generation_s": 0.0, "finish_reason": self.finish}


def test_collector_fresh_seeds_and_failed_verifier():
    backend = FakeBackend()
    args = SimpleNamespace(seed=17, group_size=2, accept_length=False)

    def reward(row, text):
        raise RuntimeError("verifier unavailable")

    g = asyncio.run(collect_group(backend, {"id": "p"}, [0], 0, 1, 4, args, reward))
    assert len({c[1] for c in backend.calls}) == 2
    assert all(c[0] == [0] for c in backend.calls)
    assert all(s["reward"] is None and s["status"] == "error" for s in g["samples"])


def test_truncation_requires_explicit_task_budget_decision():
    args = SimpleNamespace(seed=1, group_size=2, accept_length=False)
    backend = FakeBackend("length")
    reward = lambda row, text: 1.0
    g = asyncio.run(collect_group(backend, {"id": "p"}, [0], 0, 0, 0, args, reward))
    assert all(s["status"] == "truncated" and s["reward"] is None for s in g["samples"])
    args.accept_length = True
    g = asyncio.run(collect_group(backend, {"id": "p"}, [0], 0, 0, 0, args, reward))
    assert all(s["status"] == "complete" and s["reward"] == 1 for s in g["samples"])


def write_fixture(path, groups, end=True):
    manifest = {"type": "manifest", "schema": 1, "expected_groups": len(groups),
                "prompt_ids": ["p"], "trials": 1, "attempts": len(groups), "group_size": 2,
                "prompts_sha256": "fixture", "accept_length": False}
    lines = [manifest] + [dict(g, type="group") for g in groups]
    if end:
        lines.append({"type": "end", "groups": len(groups)})
    path.write_text("\n".join(json.dumps(x) for x in lines))


def test_incomplete_trace_and_no_overwrite(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_fixture(path, [group()], end=False)
    with pytest.raises(ValueError, match="Incomplete"):
        read_run(path)
    with pytest.raises(FileExistsError):
        external_output(path)


def test_repository_output_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        external_output("accidental-data.jsonl")


def test_cli_replay_roundtrip(tmp_path):
    from rtrl import main
    path, output = tmp_path / "trace.jsonl", tmp_path / "report.json"
    write_fixture(path, [group(times=(1, 4)), group(attempt=1)])
    main(["replay", "--trace", str(path), "--deadline", "2", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["resolved"] == 1 and report["attempted_groups"] == 2


def test_trace_cannot_disguise_truncation_as_completion(tmp_path):
    g = group()
    g["samples"][0]["finish_reason"] = "length"
    path = tmp_path / "truncated.jsonl"
    write_fixture(path, [g])
    with pytest.raises(ValueError, match="finish"):
        read_run(path)


def test_audit_rejects_non_timeout_censoring_before_loading_model():
    from grpo import run
    late = group(times=(4, 4))
    truncated = group(attempt=1)
    truncated["samples"][0].update(status="truncated", reward=None, finish_reason="length")
    accepted = group(attempt=2)
    args = SimpleNamespace(deadline=2, clock="ready_s", parameters="head")
    with pytest.raises(ValueError, match="censor"):
        run({}, [late, truncated, accepted], args)


def test_streamed_tokens_and_logprobs_stay_aligned(monkeypatch):
    from collect import VLLMBackend
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(RequestOutputKind=SimpleNamespace(DELTA="delta")))

    class Engine:
        async def generate(self, prompt, sampling_params, request_id):
            assert prompt == {"prompt_token_ids": [0]}
            assert sampling_params["output_kind"] == "delta" and sampling_params["temperature"] == 1
            for token, text, end in [(1, "a", False), (2, "", True)]:
                part = SimpleNamespace(token_ids=[token], text=text,
                                       logprobs=[{token: SimpleNamespace(logprob=-token)}],
                                       finish_reason="stop" if end else None)
                yield SimpleNamespace(outputs=[part], finished=end)

    backend = VLLMBackend.__new__(VLLMBackend)
    backend.engine, backend.max_tokens = Engine(), 10
    import time
    result = asyncio.run(backend.generate([0], 17, "test", time.perf_counter()))
    assert result["token_ids"] == [1, 2] and result["logprobs"] == [-1, -2]
    assert result["text"] == "a" and result["finish_reason"] == "stop"


def test_gpu_execution_cannot_fall_back_to_mac(monkeypatch):
    from collect import cuda_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="never uses MPS or CPU"):
        cuda_device()


def test_mps_audit_requires_matching_local_precision():
    from grpo import run
    args = SimpleNamespace(deadline=None, clock="ready_s", device="mps")
    with pytest.raises(ValueError, match="FP32 local trace"):
        run({"dtype": "bfloat16"}, [group()], args)


def test_audit_preserves_collected_precision(monkeypatch):
    import collect
    from grpo import run
    monkeypatch.setattr(collect, "cuda_device", lambda: "test CUDA")

    def load(*args, **kwargs):
        assert kwargs["dtype"] == torch.float32
        raise RuntimeError("precision checked before device transfer")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=load)))
    args = SimpleNamespace(deadline=None, clock="ready_s", device="cuda")
    with pytest.raises(RuntimeError, match="precision checked"):
        run({"model": "test", "revision": "fixed", "dtype": "float32"}, [group()], args)


@pytest.mark.parametrize("scale", [1e-23, 1e20])
def test_gradient_comparison_is_scale_invariant(scale):
    a = [torch.tensor([scale, scale])]
    b = [torch.tensor([scale, -scale])]
    result = compare_gradients(a, b)
    assert result["cosine"] == pytest.approx(0.0, abs=1e-12)
    assert result["relative_difference"] == pytest.approx(math.sqrt(2))
    assert result["baseline_norm"] == pytest.approx(math.sqrt(2) * scale, rel=1e-6, abs=0)


def test_masked_nonfinite_padding_cannot_poison_gradient():
    logp = torch.tensor([[-1.0, float("nan")]], requires_grad=True)
    loss = grpo_loss(logp, torch.tensor([[True, False]]), torch.tensor([1.0]))
    loss.backward()
    assert loss.item() == -1
    assert logp.grad.tolist() == [[-1, 0]]


def test_logprob_scoring_is_stable_for_large_common_offset():
    model = TinyLM()
    with torch.no_grad():
        model.weight.fill_(10000)
    actual = response_logprobs(model, [0], [1])
    assert actual.item() == pytest.approx(-math.log(3), abs=1e-6)


def test_prompt_identity_must_match_across_trials():
    other = group()
    other.update(id="p:1:0", trial=1, prompt_token_ids=[9])
    with pytest.raises(ValueError, match="prompt"):
        replay([group(), other])


@pytest.mark.parametrize("raw", [True, False])
def test_collection_extension_preserves_original_bank(tmp_path, monkeypatch, raw):
    import collect
    from rtrl import main

    class Backend(FakeBackend):
        def __init__(self, args, revision):
            super().__init__()
            self.engine = SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(collect, "VLLMBackend", Backend)
    monkeypatch.setattr(collect, "cuda_device", lambda: "H100 test double")
    monkeypatch.setattr(collect, "version", lambda name: "test")
    api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="fixed"))
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"test": 0, "[UNK]": 1}, unk_token="[UNK]")),
                                       chat_template="{{ messages[0]['content'] }}")
    factory = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=factory))
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p", "prompt": "test", "reference": "answer"})+"\n")
    first, extended = tmp_path / "first.jsonl", tmp_path / "extended.jsonl"
    common = ["collect", "--prompts", str(prompts), "--model", "test/model", "--reward", "rewards:exact_match", "--group-size", "2", "--trials", "2"]
    if raw:
        common.append("--raw-prompt")
    main(common + ["--attempts", "1", "--output", str(first)])
    original, old_groups = read_run(first)
    main(common + ["--attempts", "2", "--extend", str(first), "--output", str(extended)])
    manifest, groups = read_run(extended)
    assert manifest["attempts"] == 3 and len(groups) == 6
    assert groups[:2] == old_groups
    assert manifest["extended_from"] == original["trace_sha256"]
    assert len({s["seed"] for g in groups for s in g["samples"]}) == 12
    assert [d["baseline_id"] for d in replay(groups)["decisions"]] == [d["baseline_id"] for d in replay(old_groups)["decisions"]]
    with pytest.raises(SystemExit) as error:
        main(common + ["--seed", "18", "--attempts", "1", "--extend", str(first), "--output", str(tmp_path / "invalid.jsonl")])
    assert error.value.code == 1


def test_paired_sampling_error_matches_analytic_covariance():
    from grpo import paired_summary
    zero = [torch.tensor([0., 0.])]
    result = paired_summary(iter([("p", zero, [torch.tensor([2., 0.])]),
                                  ("p", zero, [torch.tensor([0., 2.])])]))
    assert result["difference_norm"] == pytest.approx(math.sqrt(2))
    assert result["rms_sampling_error"] == pytest.approx(math.sqrt(2))
    assert result["difference_over_rms_sampling_error"] == pytest.approx(1)


def test_sampling_error_does_not_pool_prompt_difficulty():
    from grpo import paired_summary
    zero = [torch.zeros(1)]
    pairs = [("a", zero, [torch.tensor([10.])])]*2 + [("b", zero, [torch.tensor([-10.])])]*2
    result = paired_summary(iter(pairs))
    assert result["rms_sampling_error"] == 0
    assert result["difference_norm"] == 0
    assert paired_summary(iter(pairs[:1]))["rms_sampling_error"] is None


def test_exact_probe_matches_exhaustive_groups():
    from probe import exact_binary
    p = 0.25
    probabilities = [.375, .375, .05, .2]
    scores = [-p, -p, 1-p, 1-p]
    rewards = [0, 1, 0, 1]
    admission = [1, 1, 1, .1]
    numerator = denominator = 0.0
    for group_outcomes in itertools.product(range(4), repeat=3):
        probability = math.prod(probabilities[i]*admission[i] for i in group_outcomes)
        update = sum(a*scores[i] for a,i in zip(advantages([rewards[i] for i in group_outcomes]), group_outcomes))/3
        numerator += probability*update
        denominator += probability
    exact = exact_binary(3, admission)
    assert exact["group_admission"] == pytest.approx(denominator)
    assert exact["admitted_update"] == pytest.approx(numerator/denominator)


def test_retry_coverage_matches_exhaustive_attempts():
    from probe import retry_coverage
    # Each two-response group is admitted with probability 1/4.
    resolved = sum(1 for outcomes in itertools.product([False, True], repeat=6)
                   if any(all(outcomes[i:i+2]) for i in [0, 2, 4]))/64
    assert retry_coverage(.5, 2, 3, 1)["trial_resolution"] == pytest.approx(resolved)


def test_streamed_uncertainty_matches_dense_reference():
    import numpy as np
    from grpo import paired_summary
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(2, 3, 5, 7)).astype("float32")
    pairs = ((str(p), [torch.from_numpy(a[p,t])], [torch.from_numpy(b[p,t])])
             for p in range(3) for t in range(5))
    result = paired_summary(pairs)
    expected = np.sqrt((b-a).astype("float64").var(1, ddof=1).sum()/ (5*3**2))
    assert result["rms_sampling_error"] == pytest.approx(expected, rel=1e-6)
    assert result["difference_norm"] == pytest.approx(np.linalg.norm((b-a).mean((0,1))), rel=1e-6)


def test_local_batch_records_first_eos_and_aligned_probabilities():
    from types import SimpleNamespace
    from local import generate_group

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=[2, 3])

        def generate(self, input_ids, stopping_criteria, **kwargs):
            scores = []
            for ids in ([2, 1], [0, 3]):
                input_ids = torch.cat([input_ids, torch.tensor(ids)[:, None]], dim=1)
                scores.append(torch.zeros(2, 4))
                stopping_criteria(input_ids, scores)
            return SimpleNamespace(sequences=input_ids, scores=scores)

    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: str(ids))
    samples, _ = generate_group(Model(), tokenizer, [1], 2, 17, 2)
    assert [s["token_ids"] for s in samples] == [[2], [1, 3]]
    assert all(s["finish_reason"] == "stop" for s in samples)
    assert samples[0]["generation_s"] <= samples[1]["generation_s"]
    assert samples[1]["logprobs"] == pytest.approx([-math.log(4)] * 2)


def test_local_cap_is_not_an_eos():
    from types import SimpleNamespace
    from local import generate_group

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, input_ids, stopping_criteria, **kwargs):
            ids = torch.cat([input_ids, torch.ones(2, 1, dtype=torch.long)], dim=1)
            scores = [torch.zeros(2, 3)]
            stopping_criteria(ids, scores)
            return SimpleNamespace(sequences=ids, scores=scores)

    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: str(ids))
    samples, _ = generate_group(Model(), tokenizer, [1], 2, 17, 1)
    assert all(s["finish_reason"] == "length" for s in samples)
    assert all(len(s["logprobs"]) == len(s["token_ids"]) == 1 for s in samples)


def test_integer_reward_requires_terminal_answer():
    from rewards import final_integer
    assert final_integer({"reference": 42}, "6 * 7 = 42\nFinal: 42\n") == 1
    assert final_integer({"reference": 42}, "Final: 41") == 0
    assert final_integer({"reference": 42}, "Final: 42\nBut maybe 43") == 0
    assert final_integer({"reference": 42}, "42") == 0


@pytest.mark.parametrize("text,expected", [("Final: **1,376**", 1376), (r"Answer: \boxed{-42}", -42),
                                          ("The answer is 420.", 420), ("42/7", None),
                                          ("2.42", None), ("Final: 42 but maybe", None),
                                          (r"Answer: \frac{1}{42}", None), ("Final: −42", -42),
                                          ("Final: - 42", -42), ("Final: 2^42", None)])
def test_terminal_arithmetic_answer(text, expected):
    from rewards import terminal_integer
    assert terminal_integer(text) == expected


@pytest.fixture
def local_backend(monkeypatch):
    import local
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))
    model.to = model.eval = model.requires_grad_ = lambda *a, **k: model
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [0], apply_chat_template=lambda *a, **k: [0])
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="fixed"))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model)))
    monkeypatch.setattr(local, "version", lambda name: "test")
    calls = []

    def generate(model, tokenizer, prompt, size, seed, max_tokens):
        calls.append((seed, max_tokens))
        return [{"token_ids": [1], "logprobs": [-0.5], "text": "answer", "first_token_s": 0.,
                 "generation_s": 0., "finish_reason": "stop"} for _ in range(size)], time.perf_counter()

    monkeypatch.setattr(local, "generate_group", generate)
    return local, calls


def test_local_extension_preserves_baselines_and_seeds(tmp_path, local_backend):
    from rtrl import main
    _, calls = local_backend
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "p", "prompt": "test", "reference": "answer"})+"\n")
    first, extended = tmp_path / "first.jsonl", tmp_path / "extended.jsonl"
    common = ["local", "--prompts", str(prompts), "--model", "test", "--reward", "rewards:exact_match",
              "--group-size", "2", "--trials", "2", "--max-tokens", "16"]
    main(common + ["--attempts", "1", "--output", str(first)])
    original, old_groups = read_run(first)
    main(common + ["--attempts", "2", "--extend", str(first), "--output", str(extended)])
    manifest, groups = read_run(extended)
    assert groups[:2] == old_groups
    assert manifest["attempts"] == 3 and len(groups) == 6
    assert manifest["extended_from"] == original["trace_sha256"]
    assert read_run(first)[0]["trace_sha256"] == original["trace_sha256"]
    assert len({g["seed"] for g in groups}) == 6
    assert [seed for seed, limit in calls if limit == 16] == [g["seed"] for g in groups]
    assert [d["baseline_id"] for d in replay(groups)["decisions"]] == [d["baseline_id"] for d in replay(old_groups)["decisions"]]
    for change in (["--seed", "18"], ["--raw-prompt"], ["--accept-length"]):
        invalid = tmp_path / "invalid.jsonl"
        with pytest.raises(SystemExit) as error:
            main(common + change + ["--extend", str(first), "--output", str(invalid)])
        assert error.value.code == 1 and not invalid.exists()


def test_local_verifier_failure_keeps_trace_complete(tmp_path, local_backend):
    from rtrl import main
    prompts, output = tmp_path / "prompts.jsonl", tmp_path / "trace.jsonl"
    prompts.write_text(json.dumps({"id": "p", "prompt": "test"})+"\n")  # Missing verifier reference.
    main(["local", "--prompts", str(prompts), "--model", "test", "--reward", "rewards:exact_match",
          "--trials", "1", "--output", str(output)])
    _, groups = read_run(output)
    assert all(s["status"] == "error" and s["reward"] is None for s in groups[0]["samples"])
    assert "KeyError" in groups[0]["samples"][0]["error"]


def test_shared_scoring_preserves_unobserved_and_nonfinite_rewards():
    from rollout import score_sample
    calls = []
    reward = lambda row, text: calls.append(text) or float("nan")
    sample = {"finish_reason": "length", "text": "partial"}
    score_sample(sample, {}, reward)
    assert not calls and sample["status"] == "truncated" and sample["reward"] is None
    score_sample(sample, {}, reward, accept_length=True)
    assert calls == ["partial"] and sample["status"] == "error" and sample["reward"] is None


@pytest.mark.parametrize("command", ["audit", "diagnose"])
def test_failed_report_does_not_leave_empty_output(tmp_path, monkeypatch, command):
    import rtrl
    import grpo
    import diagnose
    monkeypatch.setattr(rtrl, "read_run", lambda _: ({}, []))

    def fail(*args):
        raise ValueError("Invalid experiment")

    monkeypatch.setattr(grpo, "run", fail)
    monkeypatch.setattr(diagnose, "diagnose", fail)
    output = tmp_path / "report.json"
    args = [command, "--trace", "ignored", "--output", str(output)]
    if command == "diagnose":
        args += ["--calibration", "ignored"]
    with pytest.raises(SystemExit) as error:
        rtrl.main(args)
    assert error.value.code == 1 and not output.exists()
