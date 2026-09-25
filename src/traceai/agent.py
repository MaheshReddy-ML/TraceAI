"""Permission-bounded research command interpreter.

The agent accepts a small, explicit command language. It never evaluates Python,
opens arbitrary URLs, or invokes a shell. Model execution requires an opt-in.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from traceai.config import load_config
from traceai.engine import Experiment
from traceai.errors import ConfigurationError
from traceai.reporting import render_terminal
from traceai.storage import SQLiteRepository


class ResearchAgent:
    def __init__(
        self,
        project: Path,
        workspace: Path,
        *,
        allow_run: bool = False,
        allow_subprocess: bool = False,
    ):
        self.workspace = workspace.resolve()
        self.project = project.resolve()
        if not self.project.is_relative_to(self.workspace):
            raise ConfigurationError("Agent project must be inside its allowed workspace")
        self.repository = SQLiteRepository(self.project)
        self.allow_run = allow_run
        self.allow_subprocess = allow_subprocess

    def _path(self, name: str) -> Path:
        path = (self.workspace / name).resolve()
        if not path.is_relative_to(self.workspace) or not path.is_file():
            raise ConfigurationError("Agent can read only existing files inside its workspace")
        return path

    def execute(self, command: str) -> dict:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ConfigurationError(f"Cannot parse agent command: {exc}") from exc
        if not parts or parts[0] == "help":
            return {
                "message": "Commands: help, experiments, report ID, evidence ID CASE, config FILE, run FILE (requires --allow-run)."
            }
        if parts == ["experiments"]:
            return {"experiments": self.repository.list_experiments()}
        if len(parts) == 2 and parts[0] == "report":
            record = self.repository.get(parts[1])
            from traceai.schemas import ExperimentConfig

            config = ExperimentConfig.model_validate(record["config"])
            report = Experiment(
                config, self.project, self.repository, validate_dataset=False
            ).report(parts[1])
            return {"report": report.model_dump(mode="json"), "text": render_terminal(report)}
        if len(parts) == 3 and parts[0] == "evidence":
            items = [
                item
                for item in self.repository.get(parts[1])["evidence"]
                if item.case_id == parts[2]
            ]
            return {"evidence": [item.model_dump(mode="json") for item in items]}
        if len(parts) == 2 and parts[0] == "config":
            config = load_config(self._path(parts[1]))
            return {"config": config.model_dump(mode="json")}
        if len(parts) == 2 and parts[0] == "run":
            if not self.allow_run:
                raise ConfigurationError(
                    "Agent model execution is disabled; pass --allow-run explicitly"
                )
            if not self.allow_subprocess:
                raise ConfigurationError(
                    "Experiment provenance invokes git; pass --allow-subprocess explicitly"
                )
            config_file = self._path(parts[1])
            config = load_config(config_file)
            if config.dataset:
                self._path(config.dataset)
            for checkpoint in config.checkpoints:
                model = checkpoint.model or config.target.model
                if config.target.runtime != "mock":
                    model_path = Path(model).expanduser().resolve()
                    if not model_path.is_relative_to(self.workspace):
                        raise ConfigurationError(
                            "Agent model paths must stay inside the allowed workspace"
                        )
            experiment = Experiment(config, self.project, self.repository)
            result = experiment.run()
            return {"experiment_id": result, "status": "complete"}
        raise ConfigurationError("Unknown agent command. Use 'help' for the allowed commands")
