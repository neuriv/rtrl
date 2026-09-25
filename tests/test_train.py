"""Training protocol checks without downloads, CUDA, or external W&B writes."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import transformers

from rtrl import collect
from rtrl import train
from rtrl import train_update
from rtrl import training_rollout


def test_prompt_epochs_and_attempt_seeds_do_not_depend_on_replacements():
    rows = [{"id": str(i)} for i in range(7)]
    expected = {row["id"] for row in rows}
    for epoch in range(3):
        assert {train.prompt_at(rows, 17, epoch * 7 + i)["id"] for i in range(7)} == expected
    seeds = [train.sample_seed(17, occurrence, attempt)
             for occurrence in range(30) for attempt in (0, 1)]
    assert len(set(seeds)) == len(seeds)
    assert train.prompt_at(rows, 17, 9) == train.prompt_at(rows, 17, 9)


def test_cap_is_terminal_failure_even_with_correct_text():
    row = {"reference": "42"}
    assert train.score(row, {"text": "42", "finish_reason": "stop"}) == 1
    assert train.score(row, {"text": "42", "finish_reason": "length"}) == 0
    assert train.score(row, {"text": "41", "finish_reason": "stop"}) == 0


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    calls, runs, evaluations, loaded, artifacts = [], [], [], [], []
    decisions = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(max_position_embeddings=100)

        def to(self, *args, **kwargs):
            return self

        def save_pretrained(self, path):
            clock.now += 1

    class Tokenizer:
        def apply_chat_template(self, messages, *, return_dict=True, **kwargs):
            return {"input_ids": [1, 2]} if return_dict else [1, 2]

        def save_pretrained(self, path):
            pass

    class Optimizer:
        def __init__(self, *args, **kwargs):
            self.steps = 0

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            loaded.append(state)
            self.steps = state["steps"]

    class Run:
        def __init__(self, **kwargs):
            self.summary, self.logs, self.metrics = {}, [], []
            self.url, self.id = "https://wandb.test/run", "test-run"
            self.finished = None
            runs.append(self)

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def log_code(self, *args, **kwargs):
            pass

        def log(self, values):
            self.logs.append(copy.deepcopy(values))

        def log_artifact(self, artifact):
            pass

        def finish(self, exit_code=0):
            self.finished = exit_code

    def artifact(name, **kwargs):
        files = []
        artifacts.append({"type": kwargs["type"], "files": files})
        return SimpleNamespace(add_file=lambda path, **kw: files.append((path, kw.get("name"))))

    def git_output(command, **kwargs):
        root = Path(train.__file__).resolve().parents[1]
        if command == ["git", "rev-parse", "--show-toplevel"]:
            return str(root) + "\n"
        if command == ["git", "ls-files"]:
            assert kwargs["cwd"] == root
            return "rtrl/train.py\nenvironment/requirements-training.txt\ntests/test_train.py\n"
        return "" if command[:2] == ["git", "status"] else "a" * 40 + "\n"

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=Run, Table=lambda **kwargs: kwargs, Histogram=lambda x: x,
        plot=SimpleNamespace(scatter=lambda *args: {}),
        Artifact=artifact))
    monkeypatch.setattr(train, "version", lambda name: "test")
    monkeypatch.setattr(train.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(train.subprocess, "check_output", git_output)
    monkeypatch.setattr(collect, "cuda_device", lambda: "NVIDIA H100")
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: SimpleNamespace(sha="b" * 40))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: Model())
    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: nullcontext())

    def evaluate(model, tokenizer, rows, tokenized, args):
        assert all(isinstance(ids, list) for ids in tokenized.values())
        clock.now += 10
        evaluations.append(len(calls))
        return {"accuracy": .5, "truncation_fraction": 0, "evaluation_s": 10, "results": []}

    def generate(model, tokenizer, ids, **kwargs):
        calls.append(kwargs)
        clock.now += 2
        samples = [{"token_ids": [3, 2], "text": str(42 + slot), "finish_reason": "stop"}
                   for slot in range(kwargs["group_size"])]
        accepted = {"samples": samples, "generation_s": 2}
        return {"accepted": accepted, "discarded": None, "attempts": [accepted],
                "replaced": False, "generation_s": 2, "generated_tokens": len(samples) * 2,
                "discarded_tokens": 0, "padded_decode_tokens": len(samples) * 2}

    def update(model, optimizer, groups, device):
        assert all(sample["reward"] is not None for group in groups for sample in group["samples"])
        clock.now += 3
        stepped = decisions.pop(0) if decisions else True
        optimizer.steps += stepped
        return {"optimizer_stepped": stepped, "update_s": 3}

    monkeypatch.setattr(train, "evaluate", evaluate)
    monkeypatch.setattr(training_rollout, "retry_group", generate)
    monkeypatch.setattr(train_update, "update", update)
    for split in ("train", "eval"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(
            {"id": split, "prompt": "What is 40+2?", "reference": "42"}) + "\n")
    log = tmp_path / "experiments.md"
    log.write_text("Question: does one same-prompt retry improve learning per training second?\n")

    def launch(name, *extra):
        output = tmp_path / name
        args = train.parser().parse_args([
            "--train-prompts", str(tmp_path / "train.jsonl"),
            "--eval-prompts", str(tmp_path / "eval.jsonl"), "--experiment-log", str(log),
            "--output-dir", str(output), "--group-size", "2", "--groups-per-update", "1",
            "--max-tokens", "4", "--seconds", "100", "--max-updates", "2", "--eval-every", "1",
            *extra])
        train.run(args)
        return json.loads((output / "summary.json").read_text())

    return SimpleNamespace(launch=launch, calls=calls, runs=runs, evaluations=evaluations,
                           decisions=decisions, loaded=loaded, log=log, artifacts=artifacts)


def test_equal_update_budget_and_training_clock_exclude_evaluation(harness):
    harness.decisions.extend([False, True, True])
    result = harness.launch("baseline")
    assert (result["batches"], result["optimizer_steps"], result["groups"]) == (3, 2, 3)
    assert result["training_s"] == 15
    assert result["evaluation_s"] == 30
    assert result["checkpoint_s"] == 2
    assert harness.evaluations == [0, 2, 3]
    metrics = [log for log in harness.runs[0].logs if "eval/accuracy" in log]
    assert [(m["train/optimizer_steps"], m["train/seconds"]) for m in metrics] == [(0, 0), (1, 10), (2, 15)]
    assert "FINISHED" in harness.log.read_text()
    code_files = next(a["files"] for a in harness.artifacts if a["type"] == "code")
    root = Path(train.__file__).resolve().parents[1]
    assert code_files == [(str(root / name), name) for name in
                          ("rtrl/train.py", "environment/requirements-training.txt", "tests/test_train.py")]


def test_resume_preserves_post_checkpoint_zero_advantage_batches(harness):
    harness.decisions.extend([True, False, False])
    first = harness.launch("first", "--seconds", "15")
    checkpoint = first["checkpoint"]
    saved = json.loads((Path(checkpoint) / "state.json").read_text())["state"]
    assert saved["groups"] == first["groups"] == 3
    assert saved["training_s"] == first["training_s"] == 15
    resumed = harness.launch("resumed", "--resume", checkpoint, "--seconds", "20")
    assert resumed["optimizer_steps"] == 2
    assert resumed["groups"] == 4
    assert resumed["training_s"] == 20
    assert harness.calls[-1]["seed"] == train.sample_seed(17, 3, 0)
    assert harness.loaded == [{"steps": 1}]


def test_complete_last_batch_reports_actual_budget_overrun(harness):
    result = harness.launch("overrun", "--seconds", "1")
    assert result["training_s"] == 5
    assert result["budget_overrun_s"] == 4
    assert result["optimizer_steps"] == 1


def test_resume_rejects_changed_sampling_configuration(harness):
    first = harness.launch("first", "--max-updates", "1")
    with pytest.raises(ValueError, match="Resume configuration differs: max_tokens"):
        harness.launch("changed", "--resume", first["checkpoint"], "--max-tokens", "5")


def test_training_disables_cudnn_attention_and_rejects_old_backend_on_resume(harness):
    original = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
        first = harness.launch("backend", "--max-updates", "1")
        path = Path(first["checkpoint"]) / "state.json"
        saved = json.loads(path.read_text())
        assert saved["config"]["attention_backend"]["cudnn"] is False
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        saved["config"]["attention_backend"]["cudnn"] = True
        path.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match="Resume configuration differs: attention_backend"):
            harness.launch("changed-backend", "--resume", first["checkpoint"])
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original)
