import json

import pytest

from export_tensorboard import export, read_events


def test_only_unterminated_final_json_may_be_incomplete(tmp_path):
    path = tmp_path / "events.jsonl"
    manifest = b'{"type":"manifest"}\n'
    path.write_bytes(manifest + b'{"type":')
    events, source = read_events(tmp_path)
    assert events == [{"type": "manifest"}]
    assert source["ignored_incomplete_final_line"]
    for raw in (manifest + b'{"type":\n', manifest + b'bad\n{"type":"end"}\n'):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="Malformed event"):
            read_events(tmp_path)


def test_real_tensorboard_export_preserves_measured_axes_and_rejects_overwrite(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    source = tmp_path / "synthetic-test-only"
    source.mkdir()
    events = [{"type": "manifest", "mode": "test-only", "seed": 17},
              {"type": "eval", "optimizer_steps": 2, "training_s": 17.6, "accuracy": 0.25},
              {"type": "update", "batches": 3, "optimizer_steps": 2, "training_s": 17.6,
               "groups": 6, "replacements": 2, "generated_tokens": 100, "discarded_tokens": 20,
               "batch_mean_reward": 0.5, "batch_mixed_group_fraction": 0.75,
               "batch_grad_norm": 1.5, "batch_loss": 0, "batch_optimizer_stepped": False,
               "batch_update_s": 0.3, "batch_generation_s": 7}]
    (source / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    output = tmp_path / "snapshot"
    result = export([source], output)
    reader = EventAccumulator(str(output / result["runs"][0]["log_directory"])).Reload()
    point = reader.Scalars("eval/accuracy_by_training_seconds")[0]
    assert (point.step, point.value) == (18, 0.25)
    assert reader.Scalars("eval/accuracy_by_optimizer_updates")[0].step == 2
    point = reader.Scalars("train/wasted_token_fraction")[0]
    assert point.step == 3 and point.value == pytest.approx(0.2)
    assert reader.Scalars("train/replacement_fraction")[0].value == pytest.approx(1 / 3)
    with pytest.raises(FileExistsError):
        export([source], output)
    second = export([source], tmp_path / "snapshot-next")
    assert result["runs"] == second["runs"]
