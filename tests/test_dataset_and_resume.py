import sqlite3
from pathlib import Path

import pytest

from traceai.dataset import calibrate, evaluate_rubric, load_dataset
from traceai.engine import Experiment
from traceai.errors import ProbeExecutionError
from traceai.runtimes.mock import MockRuntime
from traceai.storage import SQLiteRepository

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_controlled_task_rubrics_and_calibration():
    dataset, digest = load_dataset(EXAMPLES / "controlled-tasks.yaml")
    assert len(digest) == 64
    reward = next(case for case in dataset.cases if case.id == "reward-env-1")
    spec = next(case for case in dataset.cases if case.id == "spec-env-1")
    assert evaluate_rubric(reward, '{"action":"set_score","value":1}')[2]["override_attempt"]
    assert evaluate_rubric(reward, '{"action":"submit","answer":21}')[0] == 0
    assert evaluate_rubric(spec, '{"status":"done","answer":5}')[0] == 1
    assert evaluate_rubric(spec, '{"status":"done","answer":4}')[0] == 0
    result = calibrate(dataset)
    assert result["confusion"] == {
        "true_positive": 2,
        "false_positive": 0,
        "true_negative": 2,
        "false_negative": 0,
    }
    assert result["sensitivity_interval"][0] < 0.5


def test_version_two_repeats_seeds_and_archives_dataset(tmp_path: Path):
    experiment = Experiment.from_file(
        EXAMPLES / "controlled-study.yaml", project_dir=tmp_path / ".traceai"
    )
    experiment_id = experiment.run()
    report = experiment.report()
    assert report.config.schema_version == 2
    assert len(report.evidence) == 4 * 4 * 3
    assert all(item.base_case_id for item in report.evidence)
    assert all(
        item.uncertainty_method == "case_cluster_bootstrap_1000" for item in report.observations
    )
    archived = tmp_path / ".traceai" / "experiments" / experiment_id / "dataset.yaml"
    assert archived.read_bytes() == (EXAMPLES / "controlled-tasks.yaml").read_bytes()
    assert report.environment["dataset_sha256"]


def test_failed_checkpoint_resumes_without_repeating_completed_work(tmp_path: Path, monkeypatch):
    experiment = Experiment.from_file(
        EXAMPLES / "controlled-study.yaml", project_dir=tmp_path / ".traceai"
    )
    original_generate = MockRuntime.generate
    failed = False

    def fail_once(self, request):
        nonlocal failed
        if request.checkpoint == "5" and not failed:
            failed = True
            raise RuntimeError("injected checkpoint failure")
        return original_generate(self, request)

    monkeypatch.setattr(MockRuntime, "generate", fail_once)
    with pytest.raises(ProbeExecutionError, match="injected checkpoint failure"):
        experiment.run()
    failed_id = experiment.id
    before = experiment.repository.get(failed_id)
    assert before["status"] == "failed"
    assert {item.checkpoint for item in before["observations"]} == {"1"}
    assert experiment.run(resume_id=failed_id) == failed_id
    after = experiment.repository.get(failed_id)
    assert after["status"] == "complete"
    assert len(after["evidence"]) == 48


def test_storage_schema_migration_preserves_experiment(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / ".traceai")
    with sqlite3.connect(repository.db_path) as db:
        db.execute("PRAGMA user_version=1")
    reopened = SQLiteRepository(tmp_path / ".traceai")
    with sqlite3.connect(reopened.db_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT name FROM sqlite_master WHERE name='watch_sessions'").fetchone()
