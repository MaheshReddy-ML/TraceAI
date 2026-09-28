"""Command-line interface for the TraceAI local service."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from contextlib import nullcontext
from pathlib import Path

from traceai.config import DEFAULT_EXPERIMENT, load_config
from traceai.dataset import DEFAULT_DATASET_TEMPLATE, calibrate, load_dataset
from traceai.engine import Experiment
from traceai.errors import ConfigurationError, TraceAIError
from traceai.guidance import report_guidance, setup_guidance
from traceai.hardware import inspect_hardware
from traceai.probes import PROBES
from traceai.reporting import render_markdown, render_terminal
from traceai.runtimes import create_runtime, discover_models, runtime_status
from traceai.schemas import Capability, CheckpointSpec, GenerationRequest, ModelSpec
from traceai.storage import SQLiteRepository
from traceai.study_setup import build_model_study
from traceai.terminal import TerminalUI
from traceai.version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceai", description="Trace how AI models learn, behave, and change."
    )
    parser.add_argument("--version", action="version", version=f"TraceAI {__version__}")
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("--quiet", action="store_true", help="suppress progress and success text")
    parser.add_argument("--verbose", action="store_true", help="show debug logs")
    parser.add_argument("--no-color", action="store_true", help="disable terminal colors")
    parser.add_argument("--no-animate", action="store_true", help="disable live terminal motion")
    parser.add_argument(
        "--project", type=Path, default=Path(".traceai"), help="project storage directory"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("models", help="discover local model snapshots and Ollama tags")
    sub.add_parser("doctor", help="diagnose local runtimes and hardware")
    guide = sub.add_parser("guide", help="get concrete next steps for a model study")
    guide.add_argument("id", nargs="?", help="saved experiment ID to investigate")
    sub.add_parser("version", help="print TraceAI version")
    init = sub.add_parser("init", help="create experiment.yaml and project storage")
    init.add_argument("--path", type=Path, default=Path("experiment.yaml"))
    init.add_argument("--guided", action="store_true", help="build a model study interactively")
    init.add_argument("--runtime", choices=["transformers", "mlx", "llama_cpp", "ollama"])
    init.add_argument("--model", help="local model path or Ollama tag")
    init.add_argument("--checkpoint", action="append", default=[], metavar="ID=MODEL")
    init.add_argument("--dataset", type=Path, help="versioned cases YAML for your model")
    init.add_argument("--probe", action="append", help="selected built-in or installed probe")
    init.add_argument("--seed", type=int, help="sampling seed (default: 42)")
    init.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    inspect = sub.add_parser("inspect", help="inspect an installed model path or identifier")
    inspect.add_argument("model")
    inspect.add_argument("--runtime", choices=["transformers", "training_state"])
    inspect.add_argument("--capability", choices=["logits", "hidden_states", "training_state"])
    inspect.add_argument("--prompt", help="prompt used for logits or hidden-state inspection")
    probe = sub.add_parser("probe", help="run one probe on a local model")
    probe.add_argument("model")
    probe.add_argument(
        "--runtime", choices=["mock", "mlx", "transformers", "ollama", "llama_cpp"], required=True
    )
    probe.add_argument("--behavior", choices=sorted(PROBES), required=True)
    probe.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    experiment = sub.add_parser("experiment", help="create, run, inspect, or compare experiments")
    exp_sub = experiment.add_subparsers(dest="action", required=True)
    create = exp_sub.add_parser("create")
    create.add_argument("path", nargs="?", type=Path, default=Path("experiment.yaml"))
    run = exp_sub.add_parser("run")
    run.add_argument("file", type=Path)
    run.add_argument(
        "--mock", action="store_true", help="replace target with deterministic mock fixture"
    )
    run.add_argument("--resume", metavar="ID", help="resume a failed or interrupted experiment")
    exp_inspect = exp_sub.add_parser("inspect")
    exp_inspect.add_argument("id")
    exp_compare = exp_sub.add_parser("compare")
    exp_compare.add_argument("first")
    exp_compare.add_argument("second")
    exp_sub.add_parser("list")
    report = sub.add_parser("report", help="render a saved experiment report")
    report.add_argument("id")
    report.add_argument("--format", choices=["terminal", "json", "markdown"], default="terminal")
    report.add_argument("--output", type=Path)
    compare = sub.add_parser("compare", help="compare two saved experiments")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--experiment", help="compare two checkpoint IDs within this experiment")
    evidence = sub.add_parser("evidence", help="inspect raw prompts and model outputs")
    evidence_sub = evidence.add_subparsers(dest="action", required=True)
    evidence_list = evidence_sub.add_parser("list")
    evidence_list.add_argument("id")
    evidence_list.add_argument("--probe")
    evidence_list.add_argument("--checkpoint")
    evidence_show = evidence_sub.add_parser("show")
    evidence_show.add_argument("id")
    evidence_show.add_argument("case_id")
    evidence_show.add_argument("--probe")
    evidence_show.add_argument("--checkpoint")
    evidence_browse = evidence_sub.add_parser("browse")
    evidence_browse.add_argument("id")
    evidence_browse.add_argument("--probe")
    evidence_browse.add_argument("--checkpoint")
    evidence_export = evidence_sub.add_parser("export")
    evidence_export.add_argument("id")
    evidence_export.add_argument("--probe")
    evidence_export.add_argument("--checkpoint")
    evidence_export.add_argument("--output", type=Path, required=True)
    config = sub.add_parser("config", help="show or validate an experiment configuration")
    config_sub = config.add_subparsers(dest="action", required=True)
    for action in ("show", "validate"):
        item = config_sub.add_parser(action)
        item.add_argument("file", type=Path)
    dataset = sub.add_parser("dataset", help="validate or calibrate a local evaluation dataset")
    dataset_sub = dataset.add_subparsers(dest="action", required=True)
    for action in ("validate", "calibrate"):
        item = dataset_sub.add_parser(action)
        item.add_argument("file", type=Path)
    dataset_template = dataset_sub.add_parser("template", help="write a synthetic cases template")
    dataset_template.add_argument("--output", type=Path, required=True)
    watch = sub.add_parser("watch", help="evaluate stable checkpoints as they appear")
    watch.add_argument("directory", type=Path)
    watch.add_argument("--config", type=Path, required=True)
    watch.add_argument("--once", action="store_true", help="process the next stable batch and exit")
    watch.add_argument(
        "--stable-for", type=float, default=2.0, help="seconds a checkpoint must remain unchanged"
    )
    watch.add_argument("--interval", type=float, default=1.0, help="poll interval in seconds")
    watch.add_argument("--timeout", type=float, default=30.0, help="maximum wait for --once")
    agent = sub.add_parser("agent", help="permission-bounded research command interpreter")
    agent.add_argument(
        "instruction", nargs="?", help="one command; omit for an interactive session"
    )
    agent.add_argument("--workspace", type=Path, default=Path.cwd())
    agent.add_argument("--allow-run", action="store_true", help="permit model execution")
    agent.add_argument(
        "--allow-subprocess",
        action="store_true",
        help="permit git provenance subprocesses during runs",
    )
    jobs = sub.add_parser("jobs", help="submit and inspect queued experiment jobs")
    jobs_sub = jobs.add_subparsers(dest="action", required=True)
    submit = jobs_sub.add_parser("submit")
    submit.add_argument("file", type=Path)
    jobs_sub.add_parser("list")
    retry = jobs_sub.add_parser("retry")
    retry.add_argument("id")
    worker = sub.add_parser("worker", help="serve or join an authenticated experiment queue")
    worker_sub = worker.add_subparsers(dest="action", required=True)
    coordinator = worker_sub.add_parser("serve")
    coordinator.add_argument("--host", default="127.0.0.1")
    coordinator.add_argument("--port", type=int, default=8766)
    coordinator.add_argument("--cert", type=Path)
    coordinator.add_argument("--key", type=Path)
    join = worker_sub.add_parser("run")
    join.add_argument("--server", required=True)
    join.add_argument("--workdir", type=Path, required=True)
    join.add_argument("--model-root", type=Path, required=True)
    join.add_argument("--ca-file", type=Path)
    join.add_argument("--max-model-size-gb", type=float, default=16.0)
    join.add_argument("--once", action="store_true")
    join.add_argument("--interval", type=float, default=2.0)
    artifacts = sub.add_parser("artifacts", help="export completed evidence to external storage")
    artifacts_sub = artifacts.add_subparsers(dest="action", required=True)
    upload = artifacts_sub.add_parser("upload")
    upload.add_argument("id")
    upload.add_argument("--destination", required=True, help="s3://bucket/prefix")
    dashboard = sub.add_parser("dashboard", help="serve a local read-only dashboard")
    dashboard.add_argument("--port", type=int, default=8765)
    return parser


def _emit(value, args, human: str | None = None) -> None:
    if args.quiet:
        return
    if args.json:
        print(json.dumps(value, ensure_ascii=False, default=str))
    else:
        print(human if human is not None else value)


def _create_file(path: Path, content: str = DEFAULT_EXPERIMENT) -> None:
    if path.exists():
        raise ConfigurationError(f"{path} already exists; choose a different path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _comparison(repository: SQLiteRepository, first: str, second: str) -> dict:
    one = repository.get(first)
    two = repository.get(second)
    a = {(item.checkpoint, item.probe): item.score for item in one["observations"]}
    b = {(item.checkpoint, item.probe): item.score for item in two["observations"]}
    common = sorted(a.keys() & b.keys())
    return {
        "first": first,
        "second": second,
        "differences": [
            {
                "checkpoint": checkpoint,
                "probe": probe,
                "first_score": a[(checkpoint, probe)],
                "second_score": b[(checkpoint, probe)],
                "delta": round(b[(checkpoint, probe)] - a[(checkpoint, probe)], 4),
            }
            for checkpoint, probe in common
        ],
        "unmatched_measurements": len(a.keys() ^ b.keys()),
    }


def _run(args) -> int:
    ui = TerminalUI(no_color=args.no_color, no_animate=args.no_animate)
    repository = (
        SQLiteRepository(args.project)
        if args.command
        in {
            "init",
            "probe",
            "experiment",
            "report",
            "compare",
            "evidence",
            "dashboard",
            "jobs",
            "worker",
            "artifacts",
        }
        or (args.command == "guide" and args.id)
        else None
    )
    if args.command is None:
        with ui.loading("Scanning local models", enabled=not args.json and not args.quiet):
            models = discover_models()
        if not args.quiet:
            if args.json:
                _emit(
                    {
                        "version": __version__,
                        "project": str(args.project.resolve()),
                        "models": len(models),
                    },
                    args,
                )
            else:
                ui.startup(args.project, len(models))
                if ui.interactive:
                    selected = {
                        "1": ["models"],
                        "2": ["doctor"],
                        "3": ["experiment", "list"],
                        "4": ["init", "--guided"],
                        "5": ["init"],
                        "6": ["guide"],
                    }.get(ui.menu())
                    if selected:
                        global_args = ["--project", str(args.project)]
                        if args.no_color:
                            global_args.append("--no-color")
                        if args.no_animate:
                            global_args.append("--no-animate")
                        return main(global_args + selected)
    elif args.command == "version":
        _emit({"version": __version__}, args, f"TraceAI {__version__}")
    elif args.command == "models":
        with ui.loading("Scanning local models", enabled=not args.json and not args.quiet):
            models = discover_models()
        if args.json:
            _emit({"models": models}, args)
        elif not args.quiet:
            ui.models(models)
    elif args.command == "doctor":
        with ui.loading("Checking local system", enabled=not args.json and not args.quiet):
            models = discover_models()
            data = {
                "hardware": inspect_hardware(),
                "runtimes": runtime_status(models),
                "local_models": len(models),
                "project": str(args.project.resolve()),
                "storage_writable": os.access(
                    args.project if args.project.exists() else args.project.parent, os.W_OK
                ),
            }
        if args.json:
            _emit(data, args)
        elif not args.quiet:
            ui.doctor(data)
    elif args.command == "guide":
        if args.id:
            record = repository.get(args.id)
            experiment = Experiment(
                load_config_from_record(record), args.project, repository, validate_dataset=False
            )
            data = report_guidance(experiment.report(args.id), args.project)
        else:
            with ui.loading("Scanning local models", enabled=not args.json and not args.quiet):
                data = setup_guidance(discover_models(), args.project)
        if args.json:
            _emit(data, args)
        elif not args.quiet:
            ui.guide(data)
    elif args.command == "init":
        if args.path.exists():
            raise ConfigurationError(f"{args.path} already exists; choose a different path")
        if args.guided:
            if not ui.interactive:
                raise ConfigurationError("--guided needs a terminal; use --runtime and --model")
            if args.json or args.quiet:
                raise ConfigurationError(
                    "--guided uses interactive prompts; omit --json and --quiet"
                )
            if (
                args.runtime
                or args.model
                or args.checkpoint
                or args.dataset
                or args.probe
                or args.seed is not None
                or args.device != "auto"
            ):
                raise ConfigurationError("Use --guided alone, or provide model options without it")
            with ui.loading("Scanning local models"):
                models = discover_models()
            choices = ui.prompt_model_study(models)
            args.runtime = choices["runtime"]
            args.model = choices["model"]
            args.checkpoint = choices["checkpoints"]
            args.dataset = choices["dataset"]
            args.probe = choices["probes"]
            args.device = choices["device"]
        if (
            args.model
            or args.runtime
            or args.checkpoint
            or args.dataset
            or args.probe
            or args.seed is not None
            or args.device != "auto"
        ):
            if not args.model or not args.runtime:
                raise ConfigurationError("A model study needs both --runtime and --model")
            config, content = build_model_study(
                runtime=args.runtime,
                model=args.model,
                checkpoint_entries=args.checkpoint,
                dataset_path=args.dataset,
                probes=args.probe,
                seed=42 if args.seed is None else args.seed,
                device=args.device,
                output=args.path,
            )
            _create_file(args.path, content)
            command_prefix = ["traceai"]
            if args.project != Path(".traceai"):
                command_prefix.extend(["--project", str(args.project)])
            data = {
                "config": str(args.path),
                "runtime": config.target.runtime,
                "checkpoints": [item.id for item in config.checkpoints],
                "probes": config.probes,
                "dataset": config.dataset,
                "next_commands": [
                    shlex.join(command_prefix + ["config", "validate", str(args.path)]),
                    shlex.join(command_prefix + ["experiment", "run", str(args.path)]),
                ],
            }
            if args.json:
                _emit(data, args)
            elif not args.quiet:
                ui.study_created(args.path, config, args.project)
        else:
            _create_file(args.path)
            _emit(
                {"config": str(args.path), "project": str(args.project)},
                args,
                f"Created {args.path} and {args.project}",
            )
    elif args.command == "inspect":
        if args.capability:
            capability = Capability(args.capability)
            runtime_name = args.runtime or (
                "training_state" if capability == Capability.TRAINING_STATE else "transformers"
            )
            if capability != Capability.TRAINING_STATE and not args.prompt:
                raise ConfigurationError(
                    "--prompt is required for logits or hidden-state inspection"
                )
            runtime = create_runtime(ModelSpec(runtime=runtime_name, model=args.model))
            try:
                runtime.load()
                request = (
                    GenerationRequest(
                        system="",
                        prompt=args.prompt,
                        seed=0,
                        max_new_tokens=8,
                        temperature=0,
                        probe="inspection",
                        case_id="prompt",
                        checkpoint="inspection",
                    )
                    if args.prompt
                    else None
                )
                result = runtime.inspect(capability, request)
            finally:
                runtime.unload()
            data = result.model_dump(mode="json")
            _emit(data, args, json.dumps(data, indent=2, ensure_ascii=False))
            return 0
        model = args.model
        matches = [
            entry
            for entry in discover_models()
            if entry["model"] == model or entry["path"] == str(Path(model).expanduser().resolve())
        ]
        path = Path(model).expanduser()
        local_model = path.is_dir() and (path / "config.json").is_file()
        data = {
            "model": model,
            "exists": local_model,
            "matches": matches,
            "capabilities": ["outputs"] if matches or local_model else [],
        }
        _emit(data, args, json.dumps(data, indent=2))
    elif args.command == "probe":
        from traceai.schemas import ExperimentConfig

        config = ExperimentConfig(
            name=f"{args.behavior}-single-probe",
            target=ModelSpec(runtime=args.runtime, model=args.model, device=args.device),
            probes=[args.behavior],
            checkpoints=[CheckpointSpec(id="baseline")],
        )
        experiment = Experiment(config, args.project, repository)
        if not args.quiet and not args.json:
            ui.experiment_start(config)
        with (
            ui.experiment_monitor(config) if not args.quiet and not args.json else nullcontext(None)
        ) as monitor:
            experiment_id = experiment.run(progress=None if args.quiet or args.json else monitor)
        if args.json:
            _emit(
                {
                    "experiment_id": experiment_id,
                    "report": experiment.report().model_dump(mode="json"),
                },
                args,
            )
        elif not args.quiet:
            result = experiment.report()
            if ui.interactive:
                ui.report(result, args.project)
            else:
                print(render_terminal(result, args.project))
    elif args.command == "experiment":
        if args.action == "create":
            _create_file(args.path)
            _emit({"config": str(args.path)}, args, f"Created {args.path}")
        elif args.action == "run":
            config = load_config(args.file)
            if args.mock:
                config = config.model_copy(
                    update={
                        "target": ModelSpec(runtime="mock", model="deterministic-fixture"),
                        "checkpoints": [
                            checkpoint.model_copy(update={"model": None})
                            for checkpoint in config.checkpoints
                        ],
                    }
                )
            experiment = Experiment(config, args.project, repository)
            if not args.quiet and not args.json:
                ui.experiment_start(config)
            previous = (
                repository.get(args.resume)
                if args.resume and not args.quiet and not args.json
                else None
            )
            with (
                ui.experiment_monitor(config, previous)
                if not args.quiet and not args.json
                else nullcontext(None)
            ) as monitor:
                experiment_id = experiment.run(
                    progress=None if args.quiet or args.json else monitor,
                    resume_id=args.resume,
                )
            data = {
                "experiment_id": experiment_id,
                "status": "complete",
                "evidence_path": str(
                    args.project / "experiments" / experiment_id / "evidence.jsonl"
                ),
            }
            if args.json:
                _emit(data, args)
            elif not args.quiet:
                result = experiment.report()
                if ui.interactive:
                    ui.report(result, args.project)
                else:
                    print(render_terminal(result, args.project))
        elif args.action == "inspect":
            record = repository.get(args.id)
            data = {
                "id": record["id"],
                "status": record["status"],
                "error": record["error"],
                "config": record["config"],
                "environment": record["environment"],
                "observations": [item.model_dump(mode="json") for item in record["observations"]],
                "evidence_count": len(record["evidence"]),
            }
            _emit(data, args, json.dumps(data, indent=2))
        elif args.action == "compare":
            data = _comparison(repository, args.first, args.second)
            if args.json:
                _emit(data, args)
            elif not args.quiet:
                ui.comparison(data)
        elif args.action == "list":
            records = repository.list_experiments()
            if args.json:
                _emit({"experiments": records}, args)
            elif not args.quiet:
                ui.experiments(records)
    elif args.command == "report":
        record = repository.get(args.id)
        experiment = Experiment(
            load_config_from_record(record), args.project, repository, validate_dataset=False
        )
        report = experiment.report(args.id)
        format_name = "json" if args.json else args.format
        output = (
            report.model_dump_json(indent=2)
            if format_name == "json"
            else render_markdown(report)
            if format_name == "markdown"
            else render_terminal(report, args.project)
        )
        if args.output:
            if args.output.exists():
                raise ConfigurationError(f"{args.output} already exists; refusing to overwrite")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                output + ("" if output.endswith("\n") else "\n"), encoding="utf-8"
            )
            _emit({"output": str(args.output)}, args, f"Wrote {args.output}")
        elif not args.quiet:
            if format_name == "terminal" and ui.interactive:
                ui.report(report, args.project)
            else:
                print(output)
    elif args.command == "compare":
        data = (
            _checkpoint_comparison(repository, args.experiment, args.first, args.second)
            if args.experiment
            else _comparison(repository, args.first, args.second)
        )
        if args.json:
            _emit(data, args)
        elif not args.quiet:
            ui.comparison(data)
    elif args.command == "evidence":
        items = repository.get(args.id)["evidence"]
        if args.probe:
            items = [item for item in items if item.probe == args.probe]
        if args.checkpoint:
            items = [item for item in items if item.checkpoint == args.checkpoint]
        if args.action == "browse":
            if not ui.interactive or args.json:
                raise ConfigurationError("Evidence browsing requires an interactive terminal")
            ui.browse_evidence(items)
        elif args.action == "export":
            if args.output.exists():
                raise ConfigurationError(f"{args.output} already exists; refusing to overwrite")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8"
            )
            args.output.chmod(0o600)
            _emit(
                {"output": str(args.output), "records": len(items)},
                args,
                f"Exported {len(items)} evidence record(s) to {args.output}",
            )
        elif args.action == "show":
            items = [item for item in items if item.case_id == args.case_id]
            if len(items) != 1:
                raise ConfigurationError(
                    f"Expected one evidence record for {args.case_id!r}; found {len(items)}. Add --checkpoint and --probe to disambiguate"
                )
            if args.json:
                _emit(items[0].model_dump(mode="json"), args)
            elif not args.quiet:
                ui.evidence(items[0])
        elif args.json:
            _emit(
                {
                    "experiment_id": args.id,
                    "evidence": [item.model_dump(mode="json") for item in items],
                },
                args,
            )
        elif not args.quiet:
            ui.evidence_list(items, args.id)
    elif args.command == "config":
        config = load_config(args.file)
        from traceai.probes import get_probe

        for name in config.probes:
            get_probe(name)
        create_runtime(config.target)
        data = config.model_dump(mode="json")
        if args.json:
            _emit({"valid": True, "config": data}, args)
        elif not args.quiet:
            print(
                json.dumps(data, indent=2)
                if args.action == "show"
                else f"Valid experiment configuration: {args.file}"
            )
    elif args.command == "dataset":
        if args.action == "template":
            _create_file(args.output, DEFAULT_DATASET_TEMPLATE)
            _emit(
                {"output": str(args.output), "synthetic": True},
                args,
                f"Created synthetic dataset template {args.output}. Replace its cases before interpreting a model result.",
            )
        else:
            dataset, digest = load_dataset(args.file)
            if args.action == "calibrate":
                data = calibrate(dataset)
                data["sha256"] = digest
                if args.json:
                    _emit(data, args)
                elif not args.quiet:
                    ui.calibration(data)
            else:
                data = {
                    "valid": True,
                    "name": dataset.name,
                    "evaluation_cases": sum(case.split == "evaluation" for case in dataset.cases),
                    "calibration_cases": sum(case.split == "calibration" for case in dataset.cases),
                    "sha256": digest,
                }
                _emit(
                    data,
                    args,
                    f"Valid dataset: {dataset.name} · {data['evaluation_cases']} evaluation cases · {data['calibration_cases']} calibration cases",
                )
    elif args.command == "watch":
        from traceai.watcher import CheckpointWatcher

        progress = None
        if not args.quiet and not args.once:
            progress = (
                (lambda message: print(json.dumps({"event": "progress", "message": message})))
                if args.json
                else ui.progress
            )
        elif not args.quiet and not args.json:
            progress = ui.progress
        watcher = CheckpointWatcher(
            args.directory,
            args.config,
            args.project,
            stable_for=args.stable_for,
            interval=args.interval,
            progress=progress,
        )
        if args.once:
            result = watcher.run_once(timeout=args.timeout)
            _emit(
                {"experiment_id": result, "directory": str(watcher.directory)},
                args,
                f"Updated experiment {result}",
            )
        else:
            if not args.json and not args.quiet and ui.animate and sys.stdin.isatty():
                with ui.watch_monitor(watcher.repository, watcher.directory) as monitor:
                    watcher.progress = monitor
                    watcher.watch(wait=monitor.wait)
            else:
                if not args.json and not args.quiet:
                    ui.heading("checkpoint watch")
                watcher.watch()
    elif args.command == "agent":
        from traceai.agent import ResearchAgent

        agent = ResearchAgent(
            args.project,
            args.workspace,
            allow_run=args.allow_run,
            allow_subprocess=args.allow_subprocess,
        )
        if args.instruction:
            result = agent.execute(args.instruction)
            _emit(result, args, result.get("text") or json.dumps(result, indent=2))
        elif sys.stdin.isatty() and not args.json:
            ui.heading("research agent")
            ui.console.print("Bounded commands only. Type help, or quit to leave.")
            while True:
                instruction = ui.console.input("traceai › ").strip()
                if instruction in {"quit", "exit"}:
                    break
                try:
                    result = agent.execute(instruction)
                    ui.console.print(result.get("text") or json.dumps(result, indent=2))
                except TraceAIError as exc:
                    ui.error(exc)
        else:
            raise ConfigurationError("Provide an agent instruction when stdin is not interactive")
    elif args.command == "jobs":
        if args.action == "submit":
            config = load_config(args.file)
            job_id = repository.submit_job(config)
            _emit({"job_id": job_id, "status": "queued"}, args, f"Queued experiment job {job_id}")
        elif args.action == "retry":
            repository.retry_job(args.id)
            _emit({"job_id": args.id, "status": "queued"}, args, f"Requeued job {args.id}")
        else:
            jobs = repository.list_jobs()
            _emit({"jobs": jobs}, args, json.dumps(jobs, indent=2))
    elif args.command == "worker":
        from traceai.workers import RemoteWorker, serve_coordinator

        if args.action == "serve":
            serve_coordinator(
                repository, host=args.host, port=args.port, cert=args.cert, key=args.key
            )
        else:
            import time

            if args.interval <= 0:
                raise ConfigurationError("--interval must be positive")
            worker = RemoteWorker(
                args.server,
                args.workdir,
                model_root=args.model_root,
                ca_file=args.ca_file,
                max_model_size_gb=args.max_model_size_gb,
            )
            while True:
                result = worker.run_once()
                if result:
                    _emit({"experiment_id": result}, args, f"Completed experiment {result}")
                elif args.once:
                    _emit({"job": None}, args, "No queued job")
                if args.once:
                    break
                time.sleep(args.interval)
    elif args.command == "artifacts":
        from traceai.artifacts import upload_experiment

        data = upload_experiment(repository, args.id, args.destination)
        _emit(data, args, f"Uploaded {len(data['keys'])} artifact(s) to s3://{data['bucket']}")
    elif args.command == "dashboard":
        from traceai.dashboard import serve

        serve(repository, args.port)
    else:
        _parser().print_help()
    return 0


def load_config_from_record(record):
    from traceai.schemas import ExperimentConfig

    return ExperimentConfig.model_validate(record["config"])


def _checkpoint_comparison(
    repository: SQLiteRepository, experiment_id: str, first: str, second: str
) -> dict:
    record = repository.get(experiment_id)
    a = {item.probe: item.score for item in record["observations"] if item.checkpoint == first}
    b = {item.probe: item.score for item in record["observations"] if item.checkpoint == second}
    if not a or not b:
        raise ConfigurationError("Both checkpoint IDs must have observations in the experiment")
    return {
        "experiment_id": experiment_id,
        "first": first,
        "second": second,
        "differences": [
            {
                "checkpoint": f"{first} → {second}",
                "probe": probe,
                "first_score": a[probe],
                "second_score": b[probe],
                "delta": round(b[probe] - a[probe], 4),
                "evidence_count": sum(
                    item.checkpoint == second and item.probe == probe for item in record["evidence"]
                ),
            }
            for probe in sorted(a.keys() & b.keys())
        ],
        "unmatched_measurements": len(a.keys() ^ b.keys()),
    }


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Accept common flags before or after a subcommand.
    common_flags = {"--json", "--quiet", "--verbose", "--no-color", "--no-animate"}
    flags = [item for item in raw if item in common_flags]
    raw = flags + [item for item in raw if item not in common_flags]
    parser = _parser()
    args = parser.parse_args(raw)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except TraceAIError as exc:
        if args.json:
            print(
                json.dumps(
                    {"error": type(exc).__name__, "message": str(exc), "exit_code": exc.exit_code}
                ),
                file=sys.stderr,
            )
        else:
            TerminalUI(no_color=args.no_color).error(exc)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
