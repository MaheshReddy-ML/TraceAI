import json
import subprocess
import sys
from pathlib import Path


def command(*args: str, cwd: Path):
    return subprocess.run(
        [sys.executable, "-m", "traceai.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_init_run_report_and_compare(tmp_path: Path):
    project = str(tmp_path / ".traceai")
    config = str(tmp_path / "experiment.yaml")
    init = command("--project", project, "init", "--path", config, cwd=tmp_path)
    assert init.returncode == 0, init.stderr
    run = command("--project", project, "experiment", "run", config, "--json", cwd=tmp_path)
    assert run.returncode == 0, run.stderr
    experiment_id = json.loads(run.stdout)["experiment_id"]
    report = command(
        "--project", project, "report", experiment_id, "--format", "json", cwd=tmp_path
    )
    assert report.returncode == 0, report.stderr
    payload = json.loads(report.stdout)
    assert payload["experiment_id"] == experiment_id
    assert len(payload["evidence"]) == 120
    compare = command(
        "--project", project, "compare", experiment_id, experiment_id, "--json", cwd=tmp_path
    )
    assert compare.returncode == 0
    assert all(item["delta"] == 0 for item in json.loads(compare.stdout)["differences"])
    checkpoints = command(
        "--project",
        project,
        "compare",
        "1",
        "15",
        "--experiment",
        experiment_id,
        "--json",
        cwd=tmp_path,
    )
    assert checkpoints.returncode == 0, checkpoints.stderr
    assert json.loads(checkpoints.stdout)["differences"]
    evidence = command(
        "--project",
        project,
        "evidence",
        "list",
        experiment_id,
        "--json",
        cwd=tmp_path,
    )
    assert len(json.loads(evidence.stdout)["evidence"]) == 120
    plain = command(
        "--project",
        project,
        "report",
        experiment_id,
        "--no-color",
        cwd=tmp_path,
    )
    assert plain.returncode == 0 and "\x1b[" not in plain.stdout
    duplicate = command("--project", project, "init", "--path", config, cwd=tmp_path)
    assert duplicate.returncode == 2
    assert "already exists" in duplicate.stderr


def test_models_command_does_not_create_project(tmp_path: Path):
    result = command("models", "--json", cwd=tmp_path)
    assert result.returncode == 0
    assert "models" in json.loads(result.stdout)
    assert not (tmp_path / ".traceai").exists()


def test_version_and_dataset_json_are_clean(tmp_path: Path):
    version = command("version", "--json", cwd=tmp_path)
    assert version.returncode == 0
    assert json.loads(version.stdout)["version"] == "0.2.0"
    dataset = Path(__file__).parents[1] / "examples" / "controlled-tasks.yaml"
    calibration = command("dataset", "calibrate", str(dataset), "--json", cwd=tmp_path)
    assert calibration.returncode == 0, calibration.stderr
    assert json.loads(calibration.stdout)["sample_count"] == 4
