import json
from pathlib import Path

import pytest

from traceai.config import DEFAULT_EXPERIMENT
from traceai.engine import Experiment
from traceai.errors import ModelNotFoundError
from traceai.reporting import render_markdown
from traceai.storage import SQLiteRepository
from traceai.trajectory import TrajectoryAnalyzer


def test_mock_pipeline_persists_complete_evidence_and_report(tmp_path: Path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(DEFAULT_EXPERIMENT)
    experiment = Experiment.from_file(config_file)
    experiment_id = experiment.run()
    report = experiment.report()

    assert report.status == "complete"
    assert report.experiment_id == experiment_id
    assert len(report.observations) == 20
    assert len(report.evidence) == 120
    assert report.environment["config_sha256"]
    assert report.environment["source_sha256"]
    assert report.environment["execution"]["generations"] == 120
    assert any(change.delta > 0 for change in report.changes)
    assert "synthetic" in " ".join(report.limitations)
    assert len(experiment.results()) == 20

    exported = tmp_path / ".traceai" / "experiments" / experiment_id / "evidence.jsonl"
    lines = [json.loads(line) for line in exported.read_text().splitlines()]
    assert len(lines) == 120
    assert all(item["prompt"] and item["system"] and item["output"] for item in lines)
    assert all(item["experiment_id"] == experiment_id for item in lines)
    assert exported.stat().st_mode & 0o077 == 0

    reopened = SQLiteRepository(tmp_path / ".traceai")
    assert reopened.get(experiment_id)["status"] == "complete"
    assert "**OBSERVED:**" in render_markdown(report)
    assert "**UNCERTAIN:**" in render_markdown(report)
    assert TrajectoryAnalyzer().compare_checkpoints(report.observations, "1", "15")


def test_mock_run_is_deterministic_for_same_config(tmp_path: Path):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(DEFAULT_EXPERIMENT)
    first = Experiment.from_file(config_file)
    first.run()
    second = Experiment.from_file(config_file)
    second.run()
    scores = lambda experiment: [
        (item.checkpoint, item.probe, item.score) for item in experiment.results()
    ]
    assert scores(first) == scores(second)


def test_failed_load_is_recorded_without_hiding_root_cause(tmp_path: Path, monkeypatch):
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text(DEFAULT_EXPERIMENT)

    class UnavailableRuntime:
        def load(self):
            raise ModelNotFoundError("local model missing")

        def unload(self):
            pass

    monkeypatch.setattr("traceai.engine.create_runtime", lambda spec: UnavailableRuntime())
    experiment = Experiment.from_file(config_file)
    with pytest.raises(ModelNotFoundError, match="local model missing"):
        experiment.run()
    record = experiment.repository.get(experiment.id)
    assert record["status"] == "failed"
    assert record["error"] == "local model missing"
