import io
import json
import subprocess
import sys
from pathlib import Path

from traceai.engine import Experiment
from traceai.terminal import LiveExperiment, TerminalUI


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


def test_model_study_setup_uses_distinct_local_checkpoints_and_dataset(tmp_path: Path):
    before = tmp_path / "before"
    after = tmp_path / "after"
    for model in (before, after):
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
    dataset = Path(__file__).parents[1] / "examples" / "controlled-tasks.yaml"
    study = tmp_path / "my-study.yaml"
    result = command(
        "init",
        "--runtime",
        "transformers",
        "--model",
        str(before),
        "--checkpoint",
        f"before={before}",
        "--checkpoint",
        f"after={after}",
        "--dataset",
        str(dataset),
        "--path",
        str(study),
        "--json",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["checkpoints"] == ["before", "after"]
    assert payload["probes"] == ["reward_hacking", "specification_gaming"]
    validated = command("config", "validate", str(study), "--json", cwd=tmp_path)
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True
    assert "traceai experiment run" in payload["next_commands"][1]

    duplicate = command(
        "init",
        "--runtime",
        "transformers",
        "--model",
        str(before),
        "--checkpoint",
        f"before={before}",
        "--checkpoint",
        f"after={before}",
        "--path",
        str(tmp_path / "duplicate.yaml"),
        cwd=tmp_path,
    )
    assert duplicate.returncode == 2
    assert "different model paths" in duplicate.stderr
    assert not (tmp_path / "duplicate.yaml").exists()


def test_guide_is_read_only_and_report_steps_use_saved_evidence(tmp_path: Path):
    project = tmp_path / "study-store"
    setup = command("--project", str(project), "guide", "--json", cwd=tmp_path)
    assert setup.returncode == 0, setup.stderr
    assert json.loads(setup.stdout)["kind"] == "setup"
    assert not project.exists()

    fixture = Path(__file__).parents[1] / "examples" / "controlled-study.yaml"
    run = command(
        "--project", str(project), "experiment", "run", str(fixture), "--json", cwd=tmp_path
    )
    assert run.returncode == 0, run.stderr
    experiment_id = json.loads(run.stdout)["experiment_id"]
    guide = command("--project", str(project), "guide", experiment_id, "--json", cwd=tmp_path)
    assert guide.returncode == 0, guide.stderr
    result = json.loads(guide.stdout)
    assert result["kind"] == "report"
    assert result["focus"]["probe"] == "reward_hacking"
    assert any("evidence show" in step["command"] for step in result["steps"])
    assert all(str(project) in step["command"] for step in result["steps"])


def test_dataset_template_is_valid_synthetic_example_and_never_overwrites(tmp_path: Path):
    cases = tmp_path / "cases.yaml"
    created = command("dataset", "template", "--output", str(cases), "--json", cwd=tmp_path)
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["synthetic"] is True
    assert "Synthetic wiring example" in cases.read_text(encoding="utf-8")
    validated = command("dataset", "validate", str(cases), "--json", cwd=tmp_path)
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["evaluation_cases"] == 2
    calibration = command("dataset", "calibrate", str(cases), "--json", cwd=tmp_path)
    assert calibration.returncode == 0, calibration.stderr
    assert json.loads(calibration.stdout)["sample_count"] == 2
    again = command("dataset", "template", "--output", str(cases), cwd=tmp_path)
    assert again.returncode == 2
    assert "already exists" in again.stderr


def test_guided_requires_terminal_and_plain_tty_has_no_ansi(tmp_path: Path, monkeypatch):
    study = tmp_path / "study.yaml"
    guided = command("init", "--guided", "--path", str(study), cwd=tmp_path)
    assert guided.returncode == 2
    assert "needs a terminal" in guided.stderr
    assert not study.exists()

    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = TTYBuffer()
    ui = TerminalUI(no_color=True, stream=stream)
    assert ui.interactive and not ui.animate
    ui.heading("system check")
    assert "TRACEAI / SYSTEM CHECK" in stream.getvalue()
    assert "\x1b[" not in stream.getvalue()
    assert not TerminalUI(no_animate=True, stream=TTYBuffer()).animate


def test_resume_monitor_starts_from_saved_checkpoints(tmp_path: Path):
    fixture = Path(__file__).parents[1] / "examples" / "controlled-study.yaml"
    experiment = Experiment.from_file(fixture, project_dir=tmp_path / ".traceai")
    experiment_id = experiment.run()
    previous = experiment.repository.get(experiment_id)
    monitor = LiveExperiment(TerminalUI(no_color=True), experiment.config, previous)
    assert monitor.completed == 4
    assert len(monitor.scores["reward_hacking"]) == 4
    monitor("checkpoint 1: already complete, skipped")
    assert monitor.completed == 4


def test_guided_ollama_selection_uses_model_tag(monkeypatch):
    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    ui = TerminalUI(no_color=True, stream=TTYBuffer())
    answers = iter(["1", "", "", ""])
    monkeypatch.setattr(ui.console, "input", lambda _: next(answers))
    choices = ui.prompt_model_study(
        [
            {
                "runtime": "ollama",
                "model": "local-model:latest",
                "path": "http://127.0.0.1:11434",
            }
        ]
    )
    assert choices["model"] == "local-model:latest"
