"""Evidence-first next steps for developing and evaluating local models."""

from __future__ import annotations

import shlex
from pathlib import Path

from traceai.schemas import Report


def _command(project: Path, *parts: str) -> str:
    words = ["traceai"]
    if project != Path(".traceai"):
        words.extend(["--project", str(project)])
    words.extend(parts)
    return shlex.join(words)


def setup_guidance(models: list[dict], project: Path) -> dict:
    steps = [
        {
            "title": "Find a local model",
            "command": _command(project, "models"),
            "reason": "Use a discovered path or an installed Ollama tag; TraceAI does not download weights.",
        },
        {
            "title": "Prepare your evaluation cases",
            "command": _command(project, "dataset", "template", "--output", "cases.yaml"),
            "reason": "Replace every synthetic example with reserved task cases and labeled calibration outputs.",
        },
        {
            "title": "Create a model study",
            "command": _command(project, "init", "--guided"),
            "reason": "Choose one or more distinct checkpoints and attach your edited cases YAML.",
        },
        {
            "title": "Run and inspect",
            "command": _command(project, "experiment", "run", "experiment.yaml"),
            "reason": "Open the saved report and raw responses before drawing conclusions.",
        },
    ]
    return {
        "kind": "setup",
        "local_models": len(models),
        "suggested_model": models[0] if models else None,
        "steps": steps,
        "note": (
            "Use independently prepared cases and calibration before interpreting a real model run. "
            "The built-in probes are narrow output proxies."
            if models
            else "Start with traceai init for a synthetic walkthrough. Install or point TraceAI "
            "to a local model before interpreting model behavior."
        ),
    }


def report_guidance(report: Report, project: Path) -> dict:
    prefix = _command(project, "report", report.experiment_id)
    steps = [
        {
            "title": "Read the complete report",
            "command": prefix,
            "reason": "Check sample counts, intervals, changes, and limitations.",
        }
    ]
    focus = None
    if report.changes:
        change = max(report.changes, key=lambda item: abs(item.delta))
        focus = {
            "probe": change.probe,
            "from_checkpoint": change.from_checkpoint,
            "to_checkpoint": change.to_checkpoint,
            "delta": change.delta,
            "classification": change.classification,
        }
        steps.append(
            {
                "title": "Compare the measured checkpoints",
                "command": _command(
                    project,
                    "compare",
                    change.from_checkpoint,
                    change.to_checkpoint,
                    "--experiment",
                    report.experiment_id,
                ),
                "reason": "A descriptive change is a prompt to inspect cases, not a causal claim.",
            }
        )
        candidates = [
            item
            for item in report.evidence
            if item.probe == change.probe and item.checkpoint == change.to_checkpoint
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
    else:
        candidates = sorted(report.evidence, key=lambda item: item.score, reverse=True)
    if candidates:
        item = candidates[0]
        steps.append(
            {
                "title": "Inspect a raw case",
                "command": _command(
                    project,
                    "evidence",
                    "show",
                    report.experiment_id,
                    item.case_id,
                    "--probe",
                    item.probe,
                    "--checkpoint",
                    item.checkpoint,
                ),
                "reason": "Read the prompt, model response, and deterministic scoring rationale.",
            }
        )
    for checkpoint in reversed(report.config.checkpoints):
        model = checkpoint.model or report.config.target.model
        if (Path(model).expanduser() / "trainer_state.json").is_file():
            steps.append(
                {
                    "title": "Review saved training metrics",
                    "command": _command(
                        project, "inspect", model, "--capability", "training_state"
                    ),
                    "reason": "Compare recorded Trainer loss or evaluation metrics with behavior observations.",
                }
            )
            break
    steps.append(
        {
            "title": "Review more cases",
            "command": _command(project, "evidence", "browse", report.experiment_id),
            "reason": "Use repeated failures to form a hypothesis for the next training iteration.",
        }
    )
    return {
        "kind": "report",
        "experiment_id": report.experiment_id,
        "focus": focus,
        "steps": steps,
        "note": (
            "This run used a synthetic mock target; repeat with your own model and independently prepared cases."
            if report.config.target.runtime == "mock"
            else "Built-in probes are narrow output proxies. Prepare independent cases and review "
            "their rubrics before interpreting model behavior."
            if not report.config.dataset
            else "For the next checkpoint, reuse the reserved evaluation cases and compare raw "
            "evidence. These measurements do not establish intent, deception, or general safety."
        ),
    }
