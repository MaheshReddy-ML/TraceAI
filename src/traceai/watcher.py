"""Stable local checkpoint discovery and incremental experiment execution."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from traceai.config import load_config
from traceai.engine import Experiment
from traceai.errors import ConfigurationError
from traceai.schemas import CheckpointSpec, ExperimentConfig
from traceai.storage import SQLiteRepository


def checkpoint_fingerprint(path: Path) -> str | None:
    config_file = path / "config.json"
    if not config_file.is_file():
        return None
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        return None
    index_file = path / "model.safetensors.index.json"
    if index_file.is_file():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
            required = set(index["weight_map"].values())
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not required or any(not (path / filename).is_file() for filename in required):
            return None
    digest = hashlib.sha256(config_file.read_bytes())
    for file in weights:
        stat = file.stat()
        digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class CheckpointWatcher:
    def __init__(
        self,
        directory: Path,
        config_file: Path,
        project_dir: Path,
        *,
        stable_for: float = 2.0,
        interval: float = 1.0,
        progress=None,
    ):
        self.directory = directory.resolve()
        if not self.directory.is_dir():
            raise ConfigurationError(f"Checkpoint directory does not exist: {directory}")
        self.base_config = load_config(config_file.resolve())
        if self.base_config.target.runtime not in {"mock", "mlx", "transformers"}:
            raise ConfigurationError(
                "Checkpoint watching supports mock, MLX, and Transformers local directories"
            )
        if stable_for < 0 or interval <= 0:
            raise ConfigurationError("stable_for must be >= 0 and interval must be > 0")
        self.stable_for = stable_for
        self.interval = interval
        self.progress = progress
        self.repository = SQLiteRepository(project_dir)
        self.project_dir = project_dir
        self.seen: dict[str, tuple[str, float]] = {}

    def _scan(self) -> list[dict]:
        now = time.monotonic()
        stable = []
        for path in sorted(self.directory.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or not path.resolve().is_relative_to(self.directory):
                continue
            fingerprint = checkpoint_fingerprint(path)
            if fingerprint is None:
                continue
            prior = self.seen.get(path.name)
            if prior is None or prior[0] != fingerprint:
                self.seen[path.name] = (fingerprint, now)
                if self.stable_for > 0:
                    continue
            elif now - prior[1] < self.stable_for:
                continue
            stable.append(
                {"id": path.name, "path": str(path.resolve()), "fingerprint": fingerprint}
            )
        return stable

    def poll(self) -> str | None:
        stable = self._scan()
        session = self.repository.get_watch_session(str(self.directory))
        if session and ExperimentConfig.model_validate(session["config"]) != self.base_config:
            raise ConfigurationError(
                "Watch configuration changed; use a different project directory for a new watch session"
            )
        processed = session["processed"] if session else []
        by_id = {item["id"]: item for item in processed}
        new = []
        for item in stable:
            previous = by_id.get(item["id"])
            if previous and previous["fingerprint"] != item["fingerprint"]:
                raise ConfigurationError(
                    f"Checkpoint {item['id']} changed after evaluation; create a new watch session"
                )
            if previous is None:
                new.append(item)
        if not new:
            return None
        combined = processed + new
        config = self.base_config.model_copy(
            update={
                "checkpoints": [
                    CheckpointSpec(id=item["id"], model=item["path"]) for item in combined
                ]
            }
        )
        # Revalidate model/checkpoint constraints after replacing the template.
        config = ExperimentConfig.model_validate(config.model_dump())
        if self.progress:
            self.progress(
                f"Detected {len(new)} stable checkpoint(s): {', '.join(item['id'] for item in new)}"
            )
        experiment = Experiment(config, self.project_dir, self.repository)
        experiment_id = experiment.run(
            progress=self.progress,
            resume_id=session["experiment_id"] if session else None,
            append_checkpoints=bool(session),
        )
        changed_during_run = [
            item["id"]
            for item in new
            if checkpoint_fingerprint(Path(item["path"])) != item["fingerprint"]
        ]
        if changed_during_run:
            message = "Checkpoint changed while evaluating: " + ", ".join(changed_during_run)
            self.repository.finish(experiment_id, "failed", message)
            self.repository.save_watch_session(
                str(self.directory), experiment_id, self.base_config, combined
            )
            raise ConfigurationError(message)
        self.repository.save_watch_session(
            str(self.directory), experiment_id, self.base_config, combined
        )
        return experiment_id

    def run_once(self, timeout: float = 30.0) -> str:
        if timeout <= 0:
            raise ConfigurationError("timeout must be > 0")
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            result = self.poll()
            if result:
                return result
            time.sleep(self.interval)
        raise ConfigurationError(
            f"No new stable checkpoint found in {self.directory} within {timeout:.1f}s"
        )

    def watch(self, wait=None) -> None:
        if self.progress:
            self.progress(f"Watching {self.directory}. Press Ctrl-C to stop.")
        while True:
            result = self.poll()
            if result and self.progress:
                self.progress(f"Updated experiment {result}")
            if (wait or time.sleep)(self.interval) is False:
                return
