"""One place for TraceAI's human terminal presentation.

Structured JSON and redirected text never pass through this layer.
"""

from __future__ import annotations

import os
import re
import select
import sys
from collections import defaultdict
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from traceai.errors import (
    ConfigurationError,
    ModelNotFoundError,
    RuntimeUnavailableError,
    StorageError,
    TraceAIError,
)
from traceai.schemas import Evidence, Report

THEME = {
    "trace": "bold cyan",
    "heading": "bold bright_white",
    "value": "bright_white",
    "muted": "dim",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "accent": "magenta",
}


class TerminalUI:
    def __init__(self, *, no_color: bool = False, stream=None):
        stream = stream or sys.stdout
        self.interactive = bool(stream.isatty())
        self.no_color = no_color or bool(os.environ.get("NO_COLOR"))
        self.console = Console(
            file=stream,
            no_color=self.no_color,
            highlight=False,
            markup=False,
            soft_wrap=not self.interactive,
        )
        self.error_console = Console(
            file=sys.stderr,
            no_color=self.no_color,
            highlight=False,
            markup=False,
            soft_wrap=not bool(sys.stderr.isatty()),
        )

    def heading(self, section: str) -> None:
        if self.interactive:
            self.console.print(Text(f"◈ TRACEAI / {section.upper()}", style=THEME["trace"]))
            self.console.print()
        else:
            self.console.print(f"TRACEAI / {section.upper()}")

    def startup(self, project: Path, model_count: int) -> None:
        if not self.interactive:
            self.console.print("TraceAI — AI behavior observatory")
            self.console.print("Trace how AI models learn, behave, and change.")
            self.console.print(f"Workspace: {project.resolve()}")
            self.console.print(f"Local text models found: {model_count}")
            self.console.print("Try: traceai models | traceai doctor | traceai experiment list")
            return
        title = Text("◈ TRACEAI", style=THEME["trace"])
        body = Group(
            title,
            Text("AI BEHAVIOR OBSERVATORY", style=THEME["heading"]),
            Text("Trace how AI models learn, behave, and change.", style=THEME["muted"]),
        )
        self.console.print(Panel(body, border_style="cyan", padding=(1, 2)))
        self.console.print(Text("  Workspace  ", style=THEME["muted"]) + str(project.resolve()))
        self.console.print(
            Text("  Models     ", style=THEME["muted"]) + f"{model_count} cached or installed"
        )
        self.console.print(Text("  ● Ready", style=THEME["success"]) + " for local experiments\n")

    def models(self, models: list[dict]) -> None:
        self.heading("local models")
        if not models:
            self.console.print(
                "No compatible local text models discovered. Mock experiments are available."
            )
            return
        if not self.interactive:
            for model in models:
                size = model.get("size_bytes")
                display_size = f"{size / (1024**3):.1f} GiB" if size is not None else "unknown size"
                self.console.print(
                    f"{model['runtime']:<13} {model['model']} ({display_size})\n  {model['path']}"
                )
            self.console.print(f"{len(models)} local model(s) discovered.")
            return
        table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"], expand=False)
        table.add_column("MODEL", overflow="fold")
        table.add_column("RUNTIME")
        table.add_column("WEIGHTS", justify="right")
        table.add_column("STATUS")
        for model in models:
            size = model.get("size_bytes")
            display_size = f"{size / (1024**3):.1f} GiB" if size is not None else "—"
            table.add_row(
                model["model"],
                model["runtime"].upper(),
                display_size,
                "cached" if model["runtime"] != "ollama" else "installed",
            )
        self.console.print(table)
        self.console.print(
            f"{len(models)} local model(s) discovered. Use the path from --json in an experiment."
        )

    def doctor(self, data: dict) -> None:
        self.heading("system check")
        hardware = data["hardware"]
        rows = [
            ("Python", hardware["python"], None),
            ("Platform", f"{hardware['system']} · {hardware['machine']}", None),
            (
                "Memory",
                f"{hardware['memory_bytes'] / 1024**3:.1f} GiB"
                if hardware["memory_bytes"]
                else "unknown",
                None,
            ),
            (
                "CUDA",
                hardware.get("cuda_device") or "not available",
                None
                if hardware.get("cuda_available")
                else "Requires a CUDA-capable GPU and PyTorch build",
            ),
        ]
        for name, available in data["runtimes"].items():
            fix = (
                None
                if available
                else (
                    "pip install 'traceai-local[mlx]'"
                    if name == "mlx"
                    else "pip install 'traceai-local[transformers]'"
                    if name == "transformers"
                    else "pip install 'traceai-local[llama_cpp]'"
                    if name == "llama_cpp"
                    else "Start Ollama and install a model with ollama pull <name>"
                )
            )
            rows.append((name.upper(), "available" if available else "not available", fix))
        rows.append(("Local models", str(data["local_models"]), None))
        rows.append(
            (
                "Storage",
                data["project"],
                None if data["storage_writable"] else "Choose a writable --project directory",
            )
        )
        for label, value, fix in rows:
            symbol = "✓" if fix is None else "○"
            self.console.print(f"  {symbol} {label:<16} {value}")
            if fix:
                self.console.print(Text(f"      Optional setup: {fix}", style=THEME["muted"]))
        self.console.print()
        self.console.print("Mock experiments are ready without a model or API key.")

    def experiment_start(self, config) -> None:
        self.heading("experiment")
        rows = [
            ("Experiment", config.name),
            ("Target", config.target.model),
            ("Runtime", config.target.runtime.upper()),
            ("Mode", "SYNTHETIC" if config.target.runtime == "mock" else "REAL"),
            ("Probes", str(len(config.probes))),
            ("Checkpoints", str(len(config.checkpoints))),
        ]
        for label, value in rows:
            self.console.print(f"  {label:<14} {value}")
        self.console.print(Rule(style="dim") if self.interactive else "─" * 52)

    def progress(self, message: str) -> None:
        self.console.print(message)

    def experiment_monitor(self, config):
        return LiveExperiment(self, config)

    def watch_monitor(self, repository, directory: Path):
        return LiveWatch(self, repository, directory)

    def report(self, report: Report) -> None:
        self.heading("report")
        self.console.print(f"  Experiment   {report.experiment_id}")
        self.console.print(
            f"  Target       {report.config.target.runtime}/{report.config.target.model}"
        )
        self.console.print(f"  Status       {report.status}")
        self.console.print("  Scope        output-only · fixed prompt suite")
        table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"], show_lines=False)
        table.add_column("CHECKPOINT")
        table.add_column("PROBE")
        table.add_column("FAILURE RATE", justify="right")
        table.add_column("CASES", justify="right")
        for item in report.observations:
            table.add_row(item.checkpoint, item.probe, f"{item.score:.1%}", str(item.sample_count))
        self.console.print(table)
        if report.changes:
            self.console.print(
                Panel(
                    "Observed adjacent score changes reached the configured descriptive threshold. Inspect the evidence before interpreting them.",
                    title="BEHAVIORAL SIGNALS",
                    border_style=THEME["warning"],
                )
                if self.interactive
                else "Observed adjacent changes:"
            )
            for item in report.changes:
                self.console.print(
                    f"  {item.probe}: {item.from_checkpoint} → {item.to_checkpoint}, {item.delta:+.1%} ({item.classification})"
                )
        else:
            self.console.print("No adjacent difference reached the descriptive threshold.")
        self.console.print(
            f"\n{len(report.evidence)} raw evidence records. Inspect with traceai evidence list {report.experiment_id}."
        )
        self.console.print("Limitations:")
        for item in report.limitations:
            self.console.print(f"  • {item}")

    def evidence(self, item: Evidence) -> None:
        self.heading("evidence")
        self.console.print(f"  Probe       {item.probe}")
        self.console.print(f"  Checkpoint  {item.checkpoint}")
        self.console.print(f"  Case        {item.case_id}")
        self.console.print(f"  Score       {item.score:.2f} · {item.rationale}")
        self.console.print(f"  Evaluator   {item.evaluator}")
        if item.evaluation:
            for key, value in item.evaluation.items():
                self.console.print(f"  {key:<11} {value}")
        if self.interactive:
            self.console.print(
                Panel(Text(item.system), title="SYSTEM INSTRUCTION", border_style="dim")
            )
            self.console.print(Panel(Text(item.prompt), title="PROMPT", border_style="cyan"))
            self.console.print(
                Panel(Text(item.output), title="MODEL RESPONSE", border_style="magenta")
            )
        else:
            self.console.print(
                f"SYSTEM INSTRUCTION\n{item.system}\nPROMPT\n{item.prompt}\nMODEL RESPONSE\n{item.output}"
            )

    def browse_evidence(self, items: list[Evidence]) -> None:
        if not items:
            self.console.print("No evidence matches the selected filters.")
            return
        index = 0
        while True:
            self.evidence(items[index])
            self.console.print(
                f"Record {index + 1}/{len(items)} · [N] Next  [P] Previous  [Q] Quit"
            )
            choice = self.console.input("evidence › ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                return
            if choice in {"n", "next"}:
                index = min(len(items) - 1, index + 1)
            elif choice in {"p", "previous"}:
                index = max(0, index - 1)

    def comparison(self, data: dict) -> None:
        self.heading("comparison")
        differences = data["differences"]
        if not differences:
            self.console.print(
                "No matching measurements. Compare runs with common checkpoint and probe IDs."
            )
            return
        table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"])
        for name in ("CHECKPOINT", "PROBE", "FIRST", "SECOND", "CHANGE"):
            table.add_column(
                name, justify="right" if name in {"FIRST", "SECOND", "CHANGE"} else "left"
            )
        for item in differences:
            table.add_row(
                item["checkpoint"],
                item["probe"],
                f"{item['first_score']:.2f}",
                f"{item['second_score']:.2f}",
                f"{item['delta']:+.2f}",
            )
        self.console.print(table)
        biggest = max(differences, key=lambda item: abs(item["delta"]))
        self.console.print(
            f"Largest observed change: {biggest['probe']} at {biggest['checkpoint']} ({biggest['delta']:+.2f})."
        )

    def experiments(self, records: list[dict]) -> None:
        self.heading("experiments")
        if not records:
            self.console.print(
                "No experiments yet. Run traceai init, then traceai experiment run experiment.yaml."
            )
            return
        if not self.interactive:
            for row in records:
                self.console.print(
                    f"{row['id']}  {row['status']:<8}  {row['name']}  {row['created_at']}"
                )
            return
        table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"])
        for name in ("ID", "STATUS", "NAME", "CREATED"):
            table.add_column(name)
        for row in records:
            table.add_row(row["id"], row["status"], row["name"], row["created_at"])
        self.console.print(table)

    def evidence_list(self, items: list[Evidence], experiment_id: str) -> None:
        self.heading("evidence")
        self.console.print(f"Experiment {experiment_id} · {len(items)} raw record(s)")
        if not items:
            return
        if not self.interactive:
            for item in items:
                self.console.print(
                    f"{item.checkpoint:<12} {item.probe:<22} {item.case_id:<25} score={item.score:.2f}  {item.rationale}"
                )
            return
        table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"])
        for name in ("CHECKPOINT", "PROBE", "CASE", "SCORE", "REASON"):
            table.add_column(name)
        for item in items:
            table.add_row(
                item.checkpoint, item.probe, item.case_id, f"{item.score:.2f}", item.rationale
            )
        self.console.print(table)
        self.console.print(
            "Use traceai evidence show ID CASE --checkpoint CHECKPOINT for a raw response."
        )

    def calibration(self, data: dict) -> None:
        self.heading("rubric calibration")
        self.console.print(f"  Dataset         {data['dataset']}")
        self.console.print(f"  Labeled outputs {data['sample_count']}")
        counts = data["confusion"]
        self.console.print(
            f"  TP {counts['true_positive']} · FP {counts['false_positive']} · TN {counts['true_negative']} · FN {counts['false_negative']}"
        )
        for name in ("sensitivity", "specificity"):
            value = data[name]
            interval = data[f"{name}_interval"]
            display = (
                f"{value:.1%} (approx. 95% interval {interval[0]:.1%}–{interval[1]:.1%})"
                if value is not None
                else "not estimable"
            )
            self.console.print(f"  {name.title():<14} {display}")
        self.console.print(f"\n{data['scope']}")

    def error(self, error: TraceAIError) -> None:
        suggestion = _suggestion(error)
        if self.interactive:
            body = Group(
                Text(str(error)), Text(f"Suggested fix: {suggestion}", style=THEME["muted"])
            )
            self.error_console.print(
                Panel(body, title=type(error).__name__, border_style=THEME["error"])
            )
        else:
            self.error_console.print(
                f"{type(error).__name__}: {error}\nSuggested fix: {suggestion}"
            )


def _suggestion(error: TraceAIError) -> str:
    if isinstance(error, ModelNotFoundError):
        return "Run traceai models and use a cached model path or installed Ollama tag."
    if isinstance(error, RuntimeUnavailableError):
        return "Run traceai doctor and install or start the selected local runtime."
    if isinstance(error, ConfigurationError):
        return "Check the experiment YAML with traceai config validate <file>."
    if isinstance(error, StorageError):
        return "Check the --project path and local filesystem permissions."
    return "Inspect the error and rerun with --verbose for details."


SPARKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    return "".join(SPARKS[min(7, max(0, round(value * 7)))] for value in values)


class LiveExperiment:
    """Render completed measurements as they arrive; never invent progress percentages."""

    def __init__(self, ui: TerminalUI, config):
        self.ui = ui
        self.config = config
        self.events: list[str] = []
        self.scores: dict[str, list[float]] = defaultdict(list)
        self.completed = 0
        self.current = "Preparing experiment"
        self.live = None

    def __enter__(self):
        if self.ui.interactive:
            self.live = Live(
                self._render(), console=self.ui.console, refresh_per_second=4, transient=False
            )
            self.live.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.live:
            self.live.stop()
        return False

    def __call__(self, message: str) -> None:
        if not self.ui.interactive:
            self.ui.progress(message)
            return
        self.events.append(message)
        self.events = self.events[-4:]
        if message.startswith("checkpoint ") and ": " in message:
            self.current = message.split(": ", 1)[0]
        if message.startswith("checkpoint complete: "):
            self.completed += 1
        match = re.fullmatch(r"\s+([a-z_]+): ([0-9.]+) failure rate from ([0-9]+) cases", message)
        if match:
            self.scores[match.group(1)].append(float(match.group(2)))
        self.live.update(self._render())

    def _render(self):
        header = Text("◈ TRACEAI  /  LIVE EXPERIMENT", style=THEME["trace"])
        summary = Text(
            f"{self.config.name}  ·  {self.config.target.runtime.upper()}  ·  "
            f"{self.completed}/{len(self.config.checkpoints)} checkpoints complete"
        )
        table = Table(box=box.SIMPLE, show_header=True, expand=True)
        table.add_column("PROBE")
        table.add_column("TRAJECTORY")
        table.add_column("LATEST", justify="right")
        for probe in self.config.probes:
            values = self.scores[probe]
            table.add_row(
                probe,
                sparkline(values) if values else "—",
                f"{values[-1]:.0%}" if values else "pending",
            )
        events = Text(
            "\n".join(self.events[-3:]) or "Waiting for first measurement…", style=THEME["muted"]
        )
        return Panel(
            Group(header, summary, Text(self.current, style=THEME["heading"]), table, events),
            border_style="cyan",
        )


class LiveWatch:
    """Persistent checkpoint view backed only by saved observations."""

    def __init__(self, ui: TerminalUI, repository, directory: Path):
        self.ui = ui
        self.repository = repository
        self.directory = directory
        self.message = "Waiting for a stable checkpoint…"
        self.experiment_id = None
        self.observations = []
        self.checkpoints = []
        self.live = None
        self.original_tty = None

    def __enter__(self):
        if self.ui.interactive:
            import termios
            import tty

            self.original_tty = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            self.live = Live(
                self._render(), console=self.ui.console, refresh_per_second=2, transient=False
            )
            self.live.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.live:
            self.live.stop()
        if self.original_tty is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_tty)
        return False

    def __call__(self, message: str) -> None:
        if not self.ui.interactive:
            self.ui.progress(message)
            return
        self.message = message
        if message.startswith("Updated experiment "):
            self.experiment_id = message.removeprefix("Updated experiment ")
            record = self.repository.get(self.experiment_id)
            self.observations = record["observations"]
            self.checkpoints = [item["id"] for item in record["config"]["checkpoints"]]
        self.live.update(self._render())

    def wait(self, seconds: float) -> None:
        if not self.ui.interactive:
            import time

            time.sleep(seconds)
            return
        readable, _, _ = select.select([sys.stdin], [], [], seconds)
        if readable:
            key = sys.stdin.read(1).lower()
            if key == "q":
                return False
            self._action(key)

    def _action(self, key: str) -> None:
        if not self.experiment_id:
            self.message = "No saved checkpoint yet."
        elif key == "e":
            record = self.repository.get(self.experiment_id)
            evidence = record["evidence"]
            self.message = (
                f"Latest evidence: {evidence[-1].probe}/{evidence[-1].case_id} "
                f"score {evidence[-1].score:.2f}. Run traceai evidence show "
                f"{self.experiment_id} {evidence[-1].case_id} --checkpoint "
                f"{evidence[-1].checkpoint}"
                if evidence
                else "No evidence yet."
            )
        elif key == "r":
            self.message = f"Run traceai report {self.experiment_id} for the full report."
        elif key == "c":
            self.message = (
                f"Run traceai compare {self.checkpoints[-2]} {self.checkpoints[-1]} "
                f"--experiment {self.experiment_id}"
                if len(self.checkpoints) > 1
                else "Need two checkpoints to compare."
            )
        self.live.update(self._render())

    def _render(self):
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("PROBE")
        table.add_column("TRAJECTORY")
        table.add_column("LATEST", justify="right")
        for probe in sorted({item.probe for item in self.observations}):
            by_checkpoint = {
                item.checkpoint: item.score for item in self.observations if item.probe == probe
            }
            scores = [by_checkpoint[name] for name in self.checkpoints if name in by_checkpoint]
            table.add_row(probe, sparkline(scores), f"{scores[-1]:.0%}")
        return Panel(
            Group(
                Text("◈ TRACEAI  /  CHECKPOINT WATCH", style=THEME["trace"]),
                Text(str(self.directory), style=THEME["muted"]),
                Text(f"{len(self.checkpoints)} checkpoint(s) measured"),
                table,
                Text(self.message, style=THEME["heading"]),
                Text("[E] Evidence  [C] Compare  [R] Report  [Q] Quit", style=THEME["muted"]),
            ),
            border_style="cyan",
        )
