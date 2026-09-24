import copy
import hashlib
import json

import pytest

from eval_cache import load_initial_evaluation


@pytest.fixture
def source(tmp_path):
    config = {"model": "tiny", "revision": "abc", "eval_sha256": "data", "max_tokens": 64,
              "precision": "FP32/BF16", "versions": {"torch": "1", "transformers": "2"},
              "attention_backend": {"implementation": "sdpa", "cudnn": False,
                                    "flash": True, "efficient": True, "math": True}}
    rows = [{"id": "a", "reference": "42"}, {"id": "b", "reference": "7"}]
    manifest = {"type": "manifest", **config, "code_commit": "old", "run_url": "https://wandb.test/source"}
    evaluation = {"type": "eval", "optimizer_steps": 0, "training_s": 0, "evaluation_s": 299,
                  "accuracy": 0, "results": [
                      {"id": "b", "text": "7", "finish_reason": "length", "tokens": 64, "reward": 1},
                      {"id": "a", "text": "42", "finish_reason": "stop", "tokens": 3, "reward": 0}]}
    path = tmp_path / "events.jsonl"

    def save(records=None):
        path.write_text("\n".join(json.dumps(record) for record in
                                  (records if records is not None else [manifest, evaluation])) + "\n")
        return path

    def score(row, sample):
        return float(sample["finish_reason"] == "stop" and sample["text"] == row["reference"])

    return config, rows, manifest, evaluation, save, score


def test_reuses_text_rescores_rewards_restores_order_and_records_provenance(source):
    config, rows, manifest, evaluation, save, score = source
    before = copy.deepcopy(config)
    path = save()
    result = load_initial_evaluation(path, config, rows, score)
    assert result["accuracy"] == result["truncation_fraction"] == .5
    assert result["evaluation_s"] == 0
    assert result["source_evaluation_s"] == 299
    assert [(r["id"], r["reward"]) for r in result["results"]] == [("a", 1), ("b", 0)]
    assert result["initial_evaluation_source"] == {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "code_commit": "old", "run_url": manifest["run_url"]}
    assert config == before


@pytest.mark.parametrize("key", ["model", "revision", "eval_sha256", "max_tokens", "precision", "versions", "attention_backend"])
def test_protocol_mismatch_or_missing_value_is_rejected(source, key):
    config, rows, manifest, evaluation, save, score = source
    manifest[key] = "different"
    with pytest.raises(ValueError, match=key):
        load_initial_evaluation(save(), config, rows, score)
    del manifest[key]
    with pytest.raises(ValueError, match=key):
        load_initial_evaluation(save(), config, rows, score)


@pytest.mark.parametrize("field,value", [("optimizer_steps", 1), ("training_s", .1)])
def test_noninitial_evaluation_cannot_be_used_even_if_later_record_claims_initial(source, field, value):
    config, rows, manifest, evaluation, save, score = source
    initial = copy.deepcopy(evaluation)
    evaluation[field] = value
    with pytest.raises(ValueError, match="initial evaluation"):
        load_initial_evaluation(save([manifest, evaluation, initial]), config, rows, score)


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "wrong_id", "text", "finish", "absent_eval"])
def test_partial_or_invalid_evaluation_is_rejected(source, change):
    config, rows, manifest, evaluation, save, score = source
    if change == "missing":
        evaluation["results"].pop()
    elif change == "duplicate":
        evaluation["results"][1] = evaluation["results"][0]
    elif change == "extra":
        evaluation["results"].append({"id": "c"})
    elif change == "wrong_id":
        evaluation["results"][0]["id"] = "other"
    elif change == "text":
        del evaluation["results"][0]["text"]
    elif change == "finish":
        evaluation["results"][0]["finish_reason"] = "deadline"
    path = save([manifest] if change == "absent_eval" else None)
    with pytest.raises(ValueError):
        load_initial_evaluation(path, config, rows, score)


def test_reused_cache_preserves_original_evaluation_cost(source):
    config, rows, manifest, evaluation, save, score = source
    evaluation["source_evaluation_s"] = evaluation["evaluation_s"]
    evaluation["evaluation_s"] = 0
    assert load_initial_evaluation(save(), config, rows, score)["source_evaluation_s"] == 299
