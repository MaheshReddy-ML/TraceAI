"""One place for TraceAI's human terminal presentation.

Structured JSON and redirected text never pass through this layer.
"""

from __future__ import annotations

import math
import os
import re
import select
import shlex
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
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

MENU_ITEMS = (
    ("Find models", "Cached snapshots and installed Ollama tags"),
    ("Check system", "Runtime and hardware readiness"),
    ("Past studies", "Inspect saved experiments"),
    ("Study my model", "Build a checkpoint study"),
    ("Mock walkthrough", "Create a synthetic starter config"),
    ("Development guide", "From local model to raw evidence"),
)


class SignalWave:
    """Decorative motion; never represents a model measurement."""

    def __init__(self, width: int = 64):
        self.width = width

    def __rich_console__(self, console, options):
        width = min(self.width, max(8, options.max_width))
        phase = time.monotonic() * 5
        heights = "▁▂▃▄▅▆▇█"
        samples = []
        highlights = []
        for position in range(width):
            value = 3.5 + 2.3 * math.sin(position * 0.42 - phase)
            value += 1.1 * math.sin(position * 0.16 + phase * 0.65)
            height = max(0, min(7, round(value)))
            samples.append(heights[height])
            highlights.append(height >= 4)
        line = Text("".join(samples), style="#6585c8")
        start = None
        for position, active in enumerate([*highlights, False]):
            if active and start is None:
                start = position
            elif not active and start is not None:
                line.stylize("#44d9dd", start, position)
                start = None
        yield line


class TerminalUI:
    def __init__(self, *, no_color: bool = False, no_animate: bool = False, stream=None):
        stream = stream or sys.stdout
        self.interactive = bool(stream.isatty())
        self.no_color = (
            no_color or bool(os.environ.get("NO_COLOR")) or os.environ.get("TERM") == "dumb"
        )
        motion_disabled = os.environ.get("TRACEAI_NO_ANIMATION", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.animate = bool(
            self.interactive
            and not no_animate
            and not self.no_color
            and not motion_disabled
            and os.environ.get("TERM") != "dumb"
        )
        self.console = Console(
            file=stream,
            no_color=self.no_color,
            force_terminal=False if self.no_color else None,
            highlight=False,
            markup=False,
            soft_wrap=not self.interactive,
        )
        self.error_console = Console(
            file=sys.stderr,
            no_color=self.no_color,
            force_terminal=False if self.no_color else None,
            highlight=False,
            markup=False,
            soft_wrap=not bool(sys.stderr.isatty()),
        )
        self.home_project = Path(".traceai")
        self.home_model_count = 0

    @property
    def animated_menu_available(self) -> bool:
        return bool(
            self.animate
            and os.name == "posix"
            and self.console.is_terminal
            and self.console.width >= 40
            and self.console.height >= 18
            and sys.stdin.isatty()
        )

    def heading(self, section: str) -> None:
        if self.interactive:
            self.console.print(Text(f"◈ TRACEAI / {section.upper()}", style=THEME["trace"]))
            self.console.print()
        else:
            self.console.print(f"TRACEAI / {section.upper()}")

    def loading(self, message: str, *, enabled: bool = True):
        if not self.animate or not enabled:
            return nullcontext()
        return self.console.status(
            f"◈ TRACEAI / {message.upper()}", spinner="dots", spinner_style="#66e6e8"
        )

    def startup(self, project: Path, model_count: int) -> None:
        self.home_project = project
        self.home_model_count = model_count
        if self.animated_menu_available:
            return
        if not self.interactive:
            self.console.print("TraceAI — AI behavior observatory")
            self.console.print("Trace how AI models learn, behave, and change.")
            self.console.print(f"Workspace: {project.resolve()}")
            self.console.print(f"Local text models found: {model_count}")
            self.console.print("Try: traceai guide | traceai models | traceai experiment list")
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

    def menu(self) -> str:
        if self.animated_menu_available:
            return self._animated_menu()
        table = Table.grid(padding=(0, 2))
        table.add_column(style=THEME["trace"], width=3)
        table.add_column(style=THEME["heading"])
        wide = self.console.width >= 72
        if wide:
            table.add_column(style=THEME["muted"])
        for number, (title, detail) in enumerate(MENU_ITEMS, 1):
            if wide:
                table.add_row(str(number), title, detail)
            else:
                table.add_row(str(number), title)
        self.console.print(Panel(table, title="CHOOSE A PATH", border_style="cyan"))
        return self.console.input("Select 1–6, or Q to quit  › ").strip().lower()

    def _menu_frame(self, selected: int):
        wide = self.console.width >= 88
        compact = not wide or self.console.height < 20
        intro = Text()
        intro.append("◈ TRACEAI", style="bold #66e6e8")
        intro.append("   /   MODEL BEHAVIOR LAB", style=THEME["muted"])
        hero = Panel(
            Group(
                intro,
                Text("Trace the evidence behind every checkpoint.", style=THEME["heading"]),
                SignalWave(min(64, self.console.width - 8)),
            ),
            border_style="#48b7c8",
            padding=(0, 1) if compact else (1, 2),
        )
        paths = Table.grid(expand=True, padding=(0, 1))
        paths.add_column(width=4)
        paths.add_column(ratio=2)
        if wide:
            paths.add_column(ratio=3, style=THEME["muted"])
        for index, (title, detail) in enumerate(MENU_ITEMS):
            active = index == selected
            pulse = "▌" if int(time.monotonic() * 3) % 2 else "▍"
            marker = Text(
                f"{pulse if active else ' '} {index + 1}",
                style="#66e6e8" if active else "dim",
            )
            label = Text(title, style="bold bright_white" if active else "white")
            if wide:
                paths.add_row(marker, label, Text(detail, style="#9cb3c9" if active else "dim"))
            else:
                paths.add_row(marker, label)
        path_panel = Panel(paths, title="EXPLORE  /  SELECT A PATH", border_style="#48b7c8")
        if compact:
            footer = Text(
                f"{self.home_model_count} models  ·  ↑↓ move  ·  Enter open  ·  Q quit", style="dim"
            )
            return Group(hero, path_panel, footer)
        workspace = str(self.home_project.resolve())
        home = str(Path.home())
        if workspace == home or workspace.startswith(home + os.sep):
            workspace = "~" + workspace[len(home) :]
        if len(workspace) > 34:
            workspace = "…" + workspace[-33:]
        telemetry = Panel(
            Group(
                Text("LOCAL WORKSPACE", style="#66e6e8"),
                Text(workspace, overflow="fold"),
                Text(f"\n{self.home_model_count} model(s) discovered", style="bold bright_white"),
                Text("Weights stay on your machine.", style=THEME["muted"]),
                Text("\nCURRENT PATH", style="#66e6e8"),
                Text(MENU_ITEMS[selected][0], style="bold bright_white"),
                Text(MENU_ITEMS[selected][1], style=THEME["muted"]),
            ),
            border_style="#6585c8",
            title="STATUS",
        )
        if wide:
            columns = Table.grid(expand=True)
            columns.add_column(ratio=3)
            columns.add_column(ratio=2)
            columns.add_row(path_panel, telemetry)
            body = columns
        else:
            body = Group(path_panel, telemetry)
        footer = Text("  ↑ / ↓  navigate     ENTER  open     1–6  jump     Q  quit", style="dim")
        return Group(hero, body, footer)

    def _animated_menu(self) -> str:
        import termios
        import tty

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        selected = 3
        try:
            tty.setcbreak(fd)
            with Live(
                console=self.console,
                screen=self.console.height >= 18,
                transient=True,
                refresh_per_second=12,
                get_renderable=lambda: self._menu_frame(selected),
            ):
                while True:
                    ready, _, _ = select.select([fd], [], [], 0.15)
                    if not ready:
                        continue
                    key = os.read(fd, 1)
                    if key == b"\x1b":
                        for _ in range(2):
                            continuation, _, _ = select.select([fd], [], [], 0.03)
                            if continuation:
                                key += os.read(fd, 1)
                            else:
                                break
                    key = key.decode("utf-8", errors="ignore")
                    if key in {"q", "Q", "\x03", "\x1b"}:
                        return "q"
                    if key in {"\r", "\n"}:
                        return str(selected + 1)
                    if key in {"\x1b[A", "k", "K"}:
                        selected = (selected - 1) % len(MENU_ITEMS)
                    elif key in {"\x1b[B", "j", "J"}:
                        selected = (selected + 1) % len(MENU_ITEMS)
                    elif len(key) == 1 and key in "123456":
                        return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)

    def prompt_model_study(self, models: list[dict]) -> dict:
        self.heading("new model study")
        self.console.print("Create a configuration first; no weights are loaded during setup.")
        self.console.print(
            "For your own cases, run traceai dataset template --output cases.yaml, edit it, then return."
        )
        choices = models[:8]
        if choices:
            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("#", style=THEME["trace"])
            table.add_column("RUNTIME")
            table.add_column("MODEL")
            for index, item in enumerate(choices, 1):
                table.add_row(str(index), item["runtime"], item["model"])
            self.console.print(table)
            selected = self.console.input("Choose a number, or M for a model path/tag  › ").strip()
        else:
            self.console.print("No compatible local models were discovered. Enter a path or tag.")
            selected = "m"
        if selected.isdigit() and 1 <= int(selected) <= len(choices):
            item = choices[int(selected) - 1]
            runtime = item["runtime"]
            model = item["model"] if runtime == "ollama" else item["path"]
        elif selected.lower() in {"m", "manual"}:
            runtime = (
                self.console.input("Runtime [transformers / mlx / llama_cpp / ollama]  › ")
                .strip()
                .lower()
            )
            model = self.console.input("Local path or Ollama tag  › ").strip()
        else:
            raise ConfigurationError("Choose a listed model number or M for manual setup")
        device = "auto"
        if runtime == "transformers":
            device = self.console.input("Device [auto / cpu / cuda] (auto)  › ").strip() or "auto"
        checkpoints = []
        self.console.print(
            "Add checkpoints to compare training stages. Use ID=PATH; Enter to finish."
        )
        while True:
            value = self.console.input("Checkpoint  › ").strip()
            if not value:
                break
            checkpoints.append(value)
        dataset_text = self.console.input(
            "Your versioned cases YAML (Enter for built-in output proxies)  › "
        ).strip()
        dataset = Path(dataset_text) if dataset_text else None
        probes = None
        if not dataset:
            selected_probes = self.console.input(
                "Probes, comma-separated (baseline,reward_hacking,specification_gaming)  › "
            ).strip()
            if selected_probes:
                probes = [name.strip() for name in selected_probes.split(",") if name.strip()]
        return {
            "runtime": runtime,
            "model": model,
            "checkpoints": checkpoints,
            "dataset": dataset,
            "probes": probes,
            "device": device,
        }

    def study_created(self, path: Path, config, project: Path) -> None:
        command_prefix = ["traceai"]
        if project != Path(".traceai"):
            command_prefix.extend(["--project", str(project)])
        commands = (
            shlex.join(command_prefix + ["config", "validate", str(path)]),
            shlex.join(command_prefix + ["experiment", "run", str(path)]),
        )
        if self.interactive:
            body = Group(
                Text(str(path), style=THEME["heading"]),
                Text(
                    f"{config.target.runtime} · {len(config.checkpoints)} checkpoint(s) · "
                    f"{len(config.probes)} probe(s)",
                    style=THEME["muted"],
                ),
                Text("No weights were loaded during setup.", style=THEME["muted"]),
                Text("\nNext commands", style=THEME["trace"]),
                Text("\n".join(commands)),
                Text(
                    "\nAfter the run: traceai guide ID",
                    style=THEME["muted"],
                ),
            )
            self.console.print(Panel(body, title="STUDY READY", border_style="cyan"))
        else:
            self.console.print(f"Created {path}. No weights were loaded.")
            self.console.print("\n".join(commands))
            self.console.print("After the run: traceai guide ID")

    def guide(self, data: dict) -> None:
        self.heading("model development guide")
        if data["kind"] == "setup":
            count = data["local_models"]
            self.console.print(f"  Local models found  {count}")
            if data["suggested_model"]:
                item = data["suggested_model"]
                self.console.print(f"  First option         {item['model']} · {item['runtime']}")
        else:
            focus = data["focus"]
            if focus:
                self.console.print(
                    Panel(
                        f"{focus['probe']}  ·  {focus['from_checkpoint']} → "
                        f"{focus['to_checkpoint']}  ·  {focus['delta']:+.1%}\n"
                        "Descriptive change. Inspect the raw cases before interpreting it.",
                        title="MEASUREMENT TO INVESTIGATE",
                        border_style="yellow",
                    )
                )
            else:
                self.console.print(
                    "No adjacent change passed the configured descriptive threshold."
                )
        for index, step in enumerate(data["steps"], 1):
            self.console.print(Text(f"  {index:02}  {step['title']}", style=THEME["heading"]))
            self.console.print(Text(f"      {step['command']}", style=THEME["trace"]))
            self.console.print(Text(f"      {step['reason']}", style=THEME["muted"]))
        self.console.print()
        self.console.print(Text(data["note"], style=THEME["muted"]))

    def models(self, models: list[dict]) -> None:
        self.heading("local models")
        if not models:
            message = (
                "No compatible local text models discovered. Mock experiments are available.\n"
                "Run traceai guide for the model development path."
            )
            self.console.print(Panel(message, border_style="cyan") if self.interactive else message)
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
        self.console.print(Text("Next: traceai init --guided", style=THEME["trace"]))

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
        if self.interactive:
            table = Table(box=box.SIMPLE_HEAVY, header_style=THEME["heading"], expand=True)
            table.add_column("CHECK", style=THEME["heading"], no_wrap=True)
            table.add_column("RESULT")
            table.add_column("ACTION")
            for label, value, fix in rows:
                table.add_row(label, str(value), fix or "ready")
            self.console.print(table)
            status = (
                "Mock studies can run now. Use traceai guide to create a local model study."
                if data["storage_writable"]
                else "Choose a writable --project directory before running an experiment."
            )
            self.console.print(
                Panel(
                    status,
                    title="READY TO EXPLORE" if data["storage_writable"] else "SETUP NEEDED",
                    border_style="green" if data["storage_writable"] else "yellow",
                )
            )
            return
        for label, value, fix in rows:
            symbol = "✓" if fix is None else "○"
            self.console.print(f"  {symbol} {label:<16} {value}")
            if fix:
                self.console.print(Text(f"      Optional setup: {fix}", style=THEME["muted"]))
        self.console.print()
        self.console.print(
            "Mock experiments are ready without a model or API key."
            if data["storage_writable"]
            else "Choose a writable --project directory before running an experiment."
        )

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

    def experiment_monitor(self, config, previous: dict | None = None):
        return LiveExperiment(self, config, previous)

    def watch_monitor(self, repository, directory: Path):
        return LiveWatch(self, repository, directory)

    def report(self, report: Report, project: Path = Path(".traceai")) -> None:
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
        command = ["traceai"]
        if project != Path(".traceai"):
            command.extend(["--project", str(project)])
        self.console.print(
            Text(
                f"\nNext: {shlex.join(command + ['guide', report.experiment_id])}",
                style=THEME["trace"],
            )
        )

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
        message = str(error).lower()
        if "--guided" in message or "needs a terminal" in message:
            return "Run traceai init --guided in a terminal, or pass --runtime and --model."
        if "checkpoint" in message and ("different model" in message or "id=" in message):
            return "Use distinct --checkpoint ID=MODEL entries for the model stages to compare."
        if "model" in message and ("config.json" in message or ".gguf" in message):
            return "Run traceai models and use a discovered local path."
        if "dataset" in message:
            return "Run traceai dataset validate CASES.yaml and review its evaluation cases."
        return "Check the experiment YAML with traceai config validate <file>."
    if isinstance(error, StorageError):
        return "Check the --project path and local filesystem permissions."
    return "Inspect the error and rerun with --verbose for details."


SPARKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    return "".join(SPARKS[min(7, max(0, round(value * 7)))] for value in values)


class LiveExperiment:
    """Render completed measurements as they arrive; never invent progress percentages."""

    def __init__(self, ui: TerminalUI, config, previous: dict | None = None):
        self.ui = ui
        self.config = config
        self.events: list[str] = []
        self.scores: dict[str, list[float]] = defaultdict(list)
        self.completed_ids: set[str] = set()
        if previous:
            observations = previous["observations"]
            for checkpoint in config.checkpoints:
                by_probe = {
                    item.probe: item.score
                    for item in observations
                    if item.checkpoint == checkpoint.id
                }
                if all(probe in by_probe for probe in config.probes):
                    self.completed_ids.add(checkpoint.id)
                    for probe in config.probes:
                        self.scores[probe].append(by_probe[probe])
        self.completed = len(self.completed_ids)
        self.current_checkpoint = next(
            (item.id for item in config.checkpoints if item.id not in self.completed_ids), None
        )
        self.current = "Preparing experiment"
        self.started = time.monotonic()
        self.live = None

    def __enter__(self):
        if self.ui.animate:
            self.live = Live(
                console=self.ui.console,
                refresh_per_second=12,
                transient=False,
                get_renderable=self._render,
            )
            self.live.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.live:
            if exc_type:
                self.current = "Run stopped; saved checkpoints can be resumed."
                self.live.update(self._render())
            self.live.stop()
        return False

    def __call__(self, message: str) -> None:
        self.events.append(message)
        self.events = self.events[-4:]
        if message.startswith("checkpoint complete: "):
            checkpoint = message.rsplit(": ", 1)[-1]
            self.completed_ids.add(checkpoint)
            self.completed = len(self.completed_ids)
            self.current_checkpoint = next(
                (item.id for item in self.config.checkpoints if item.id not in self.completed_ids),
                None,
            )
            self.current = f"Checkpoint {checkpoint} saved"
        elif message.startswith("checkpoint ") and message.endswith(": already complete, skipped"):
            checkpoint = message.removeprefix("checkpoint ").removesuffix(
                ": already complete, skipped"
            )
            self.completed_ids.add(checkpoint)
            self.completed = len(self.completed_ids)
            self.current_checkpoint = next(
                (item.id for item in self.config.checkpoints if item.id not in self.completed_ids),
                None,
            )
            self.current = f"Checkpoint {checkpoint} already saved"
        elif message.startswith("checkpoint ") and ": " in message:
            checkpoint = message.split(": ", 1)[0]
            self.current_checkpoint = checkpoint.removeprefix("checkpoint ")
            self.current = f"Evaluating {checkpoint}"
        elif message.startswith("model loaded: "):
            self.current = "Model loaded; evaluating cases"
        match = re.fullmatch(r"\s+([a-z_]+): ([0-9.]+) failure rate from ([0-9]+) cases", message)
        if match:
            self.scores[match.group(1)].append(float(match.group(2)))
        if self.live:
            self.live.update(self._render())
        elif (
            not self.ui.interactive or message.startswith(("model loaded:", "checkpoint ")) or match
        ):
            self.ui.progress(message)

    def _render(self):
        total = len(self.config.checkpoints)
        complete = self.completed >= total
        bar_width = 24 if self.ui.console.width >= 70 else 16
        filled = round(bar_width * min(self.completed, total) / total)
        progress = Text()
        progress.append("━" * filled, style=THEME["trace"])
        progress.append("─" * (bar_width - filled), style=THEME["muted"])
        activity = (
            Text("✓ All checkpoint measurements saved", style=THEME["success"])
            if complete
            else Spinner("dots", text=f"{self.current} · {int(time.monotonic() - self.started)}s")
        )
        rail = Text()
        for index, checkpoint in enumerate(self.config.checkpoints):
            if index:
                rail.append(" ─ ", style="dim")
            if checkpoint.id in self.completed_ids:
                rail.append(f"● {checkpoint.id}", style=THEME["success"])
            elif checkpoint.id == self.current_checkpoint:
                rail.append(f"◉ {checkpoint.id}", style=THEME["trace"])
            else:
                rail.append(f"○ {checkpoint.id}", style=THEME["muted"])
        signal = SignalWave(min(48, self.ui.console.width - 8)) if not complete else None
        if self.ui.console.width < 70:
            rail.truncate(max(20, self.ui.console.width - 6), overflow="ellipsis")
            lines = [
                Text("◈ TRACEAI / RUN", style=THEME["trace"]),
                Text(
                    f"{self.completed}/{total} checkpoints · {self.config.target.runtime.upper()}"
                ),
                rail,
                progress,
                activity,
            ]
            if signal:
                lines.append(signal)
            for probe in self.config.probes:
                values = self.scores[probe]
                value = f"{values[-1]:.0%}" if values else "pending"
                lines.append(Text(f"{probe}: {sparkline(values) or '—'}  {value}"))
            return Panel(Group(*lines), border_style="cyan")

        header = Text("◈ TRACEAI  /  LIVE EXPERIMENT", style=THEME["trace"])
        summary = Text(f"{self.config.name}  ·  {self.config.target.runtime.upper()}")
        checkpoint_count = Text(f"{self.completed}/{total} checkpoints complete")
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
            self.events[-1] if self.events else "Waiting for first measurement…",
            style=THEME["muted"],
            overflow="ellipsis",
            no_wrap=True,
        )
        return Panel(
            Group(
                header,
                summary,
                checkpoint_count,
                rail,
                progress,
                activity,
                *([signal] if signal else []),
                table,
                events,
            ),
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
        if self.ui.animate:
            import termios
            import tty

            self.original_tty = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            self.live = Live(
                console=self.ui.console,
                refresh_per_second=10,
                transient=False,
                get_renderable=self._render,
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
        if not self.ui.animate:
            self.ui.progress(message)
            return
        self.message = message
        if message.startswith("Updated experiment "):
            self.experiment_id = message.removeprefix("Updated experiment ")
            record = self.repository.get(self.experiment_id)
            self.observations = record["observations"]
            self.checkpoints = [item["id"] for item in record["config"]["checkpoints"]]
        if self.live:
            self.live.update(self._render())

    def wait(self, seconds: float) -> None:
        if not self.ui.animate:
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
        if self.live:
            self.live.update(self._render())

    def _render(self):
        if self.observations:
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
        else:
            table = Text("No saved measurements yet.", style=THEME["muted"])
        return Panel(
            Group(
                Text("◈ TRACEAI  /  CHECKPOINT WATCH", style=THEME["trace"]),
                Text(str(self.directory), style=THEME["muted"]),
                Text(f"{len(self.checkpoints)} checkpoint(s) measured"),
                SignalWave(min(48, self.ui.console.width - 8)),
                table,
                Spinner("dots", text=self.message)
                if not self.experiment_id
                else Text(self.message, style=THEME["heading"]),
                Text("[E] Evidence  [C] Compare  [R] Report  [Q] Quit", style=THEME["muted"]),
            ),
            border_style="cyan",
        )
