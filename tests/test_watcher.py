from pathlib import Path

import pytest

from traceai.config import DEFAULT_EXPERIMENT
from traceai.errors import ConfigurationError
from traceai.watcher import CheckpointWatcher, checkpoint_fingerprint


def checkpoint(root: Path, name: str):
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text('{"architectures":["FixtureForCausalLM"]}')
    (path / "model.safetensors").write_bytes(b"fixture weights")
    return path


def test_watcher_appends_stable_checkpoints_and_rejects_mutated_history(tmp_path: Path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    config = tmp_path / "experiment.yaml"
    config.write_text(DEFAULT_EXPERIMENT)
    first = checkpoint(root, "epoch-1")
    watcher = CheckpointWatcher(root, config, tmp_path / ".traceai", stable_for=0)
    experiment_id = watcher.poll()
    assert experiment_id
    assert watcher.poll() is None

    checkpoint(root, "epoch-2")
    assert watcher.poll() == experiment_id
    record = watcher.repository.get(experiment_id)
    assert record["status"] == "complete"
    assert len(record["observations"]) == 10
    assert len(record["evidence"]) == 60

    (first / "config.json").write_text('{"changed":true}')
    with pytest.raises(ConfigurationError, match="changed after evaluation"):
        watcher.poll()


def test_incomplete_sharded_checkpoint_is_not_stable(tmp_path: Path):
    path = checkpoint(tmp_path, "partial")
    (path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"missing.safetensors"}}'
    )
    assert checkpoint_fingerprint(path) is None
