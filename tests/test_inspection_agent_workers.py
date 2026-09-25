import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from traceai.agent import ResearchAgent
from traceai.artifacts import upload_experiment
from traceai.config import load_config
from traceai.engine import Experiment
from traceai.errors import ConfigurationError
from traceai.runtimes.training_state import TrainingStateRuntime
from traceai.schemas import Capability, ModelSpec
from traceai.storage import SQLiteRepository
from traceai.workers import RemoteWorker

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_training_state_inspection_is_read_only_and_reloadable(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 10,
                "epoch": 1.5,
                "best_metric": 0.7,
                "log_history": [{"step": 10, "loss": 0.3, "private_note": "excluded"}],
            }
        )
    )
    runtime = TrainingStateRuntime(ModelSpec(runtime="training_state", model=str(checkpoint)))
    result = runtime.inspect(Capability.TRAINING_STATE)
    assert result.measurements["global_step"] == 10
    assert "private_note" not in result.measurements["log_history"][0]
    runtime.unload()
    assert runtime.inspect(Capability.TRAINING_STATE).measurements["epoch"] == 1.5


def test_agent_restricts_paths_and_requires_execution_permissions(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / "study.yaml"
    config.write_text(
        (EXAMPLES / "controlled-study.yaml")
        .read_text()
        .replace("controlled-tasks.yaml", str(EXAMPLES / "controlled-tasks.yaml"))
    )
    agent = ResearchAgent(workspace / ".traceai", workspace)
    assert "Commands:" in agent.execute("help")["message"]
    with pytest.raises(ConfigurationError, match="inside its workspace"):
        agent.execute(f"config {EXAMPLES / 'controlled-study.yaml'}")
    with pytest.raises(ConfigurationError, match="--allow-run"):
        agent.execute("run study.yaml")
    with pytest.raises(ConfigurationError, match="--allow-subprocess"):
        ResearchAgent(workspace / ".traceai", workspace, allow_run=True).execute("run study.yaml")
    (workspace / "controlled-tasks.yaml").write_bytes(
        (EXAMPLES / "controlled-tasks.yaml").read_bytes()
    )
    config.write_text((EXAMPLES / "controlled-study.yaml").read_text())
    permitted = ResearchAgent(
        workspace / ".traceai", workspace, allow_run=True, allow_subprocess=True
    )
    run = permitted.execute("run study.yaml")
    assert run["status"] == "complete"
    assert len(permitted.execute("experiments")["experiments"]) == 1


def test_queue_claim_import_and_s3_export(tmp_path: Path):
    coordinator = SQLiteRepository(tmp_path / "coordinator")
    config = load_config(EXAMPLES / "controlled-study.yaml")
    job = coordinator.submit_job(config)
    assert coordinator.claim_job()["id"] == job
    assert coordinator.claim_job() is None
    worker_project = tmp_path / "worker"
    experiment = Experiment(config, worker_project)
    result_id = experiment.run()
    record = experiment.repository.get(result_id)
    coordinator.import_record(record)
    coordinator.finish_job(job, result_id=result_id)
    assert coordinator.get(result_id)["status"] == "complete"
    assert coordinator.list_jobs()[0]["result_id"] == result_id

    class FakeS3:
        def __init__(self):
            self.objects = {}

        def put_object(self, **kwargs):
            assert kwargs["ServerSideEncryption"] == "AES256"
            self.objects[kwargs["Key"]] = kwargs["Body"]

    fake = FakeS3()
    result = upload_experiment(coordinator, result_id, "s3://research/traceai", client=fake)
    assert len(result["keys"]) == 2  # Worker dataset is not copied to coordinator.
    assert b"experiment_id" in fake.objects[f"traceai/{result_id}/evidence.jsonl"]


def test_authenticated_loopback_worker_round_trip(tmp_path: Path, monkeypatch):
    token = "test-token-" + "x" * 32
    monkeypatch.setenv("TRACEAI_WORKER_TOKEN", token)
    coordinator = SQLiteRepository(tmp_path / "coordinator")
    config = load_config(EXAMPLES / "controlled-study.yaml")
    job_id = coordinator.submit_job(config)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "TRACEAI_WORKER_TOKEN": token}
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "traceai.cli",
            "--project",
            str(coordinator.project_dir),
            "worker",
            "serve",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        for _ in range(50):
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        else:
            pytest.fail("coordinator did not start")
        worker = RemoteWorker(f"http://127.0.0.1:{port}", tmp_path / "worker", model_root=EXAMPLES)
        result_id = worker.run_once()
        assert result_id
        assert coordinator.get(result_id)["status"] == "complete"
        assert coordinator.get_job(job_id)["status"] == "complete"
        assert (
            coordinator.project_dir / "experiments" / result_id / "dataset.yaml"
        ).read_bytes() == (EXAMPLES / "controlled-tasks.yaml").read_bytes()
    finally:
        server.terminate()
        server.wait(timeout=5)
