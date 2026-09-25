"""Checkpoint-by-checkpoint experiment execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

from traceai.config import load_config
from traceai.dataset import evaluate_rubric, load_dataset
from traceai.errors import ConfigurationError, ProbeExecutionError, TraceAIError
from traceai.hardware import inspect_hardware
from traceai.probes import get_probe
from traceai.runtimes import create_runtime
from traceai.schemas import (
    Evidence,
    ExperimentConfig,
    GenerationRequest,
    ModelSpec,
    Observation,
    Report,
)
from traceai.storage import ExperimentRepository, SQLiteRepository, project_run_lock
from traceai.trajectory import TrajectoryAnalyzer, clustered_bootstrap_interval, wilson_interval
from traceai.version import __version__

log = logging.getLogger(__name__)


def environment_record(config: ExperimentConfig, cwd: Path) -> dict:
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        git_revision = None
        git_dirty = None
    dependencies = {}
    for package in ("traceai-local", "pydantic", "PyYAML", "transformers", "mlx-lm", "torch"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    model_path = Path(config.target.model).expanduser()
    model_config_hash = None
    if model_path.is_dir() and (model_path / "config.json").is_file():
        model_config_hash = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
    checkpoint_config_hashes = {}
    for checkpoint in config.checkpoints:
        path = Path(checkpoint.model or config.target.model).expanduser()
        config_file = path / "config.json"
        checkpoint_config_hashes[checkpoint.id] = (
            hashlib.sha256(config_file.read_bytes()).hexdigest() if config_file.is_file() else None
        )
    package_dir = Path(__file__).parent
    source_digest = hashlib.sha256()
    for source in sorted(package_dir.rglob("*.py")):
        source_digest.update(str(source.relative_to(package_dir)).encode())
        source_digest.update(source.read_bytes())
    return {
        "traceai_version": __version__,
        "git_revision": git_revision,
        "git_worktree_dirty": git_dirty,
        "source_sha256": source_digest.hexdigest(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": inspect_hardware(),
        "dependencies": dependencies,
        "model_config_sha256": model_config_hash,
        "checkpoint_config_sha256": checkpoint_config_hashes,
        "seed": config.seed,
        "config_sha256": hashlib.sha256(config.model_dump_json().encode()).hexdigest(),
        "probe_versions": {name: get_probe(name).version for name in config.probes},
    }


class Experiment:
    """Run a validated experiment using the same service as the CLI.

    Example: Experiment.from_file('experiment.yaml').run()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        project_dir: Path | str = ".traceai",
        repository: ExperimentRepository | None = None,
        analyzer: TrajectoryAnalyzer | None = None,
        validate_dataset: bool = True,
    ):
        if validate_dataset:
            for name in config.probes:
                get_probe(name)
        self.config = config
        self.dataset = None
        self.dataset_hash = None
        if config.dataset and validate_dataset:
            self.dataset, self.dataset_hash = load_dataset(Path(config.dataset))
            for name in config.probes:
                if not any(
                    case.probe == name and case.split == "evaluation" for case in self.dataset.cases
                ):
                    raise ConfigurationError(f"Dataset has no evaluation cases for probe {name!r}")
        self.project_dir = Path(project_dir)
        self.repository = repository or SQLiteRepository(self.project_dir)
        self.analyzer = analyzer or TrajectoryAnalyzer()
        self.id: str | None = None

    @classmethod
    def from_file(cls, file: Path | str, project_dir: Path | str | None = None) -> Experiment:
        path = Path(file).resolve()
        return cls(load_config(path), project_dir or path.parent / ".traceai")

    def run(
        self, progress=None, resume_id: str | None = None, append_checkpoints: bool = False
    ) -> str:
        with project_run_lock(self.project_dir):
            return self._run_unlocked(progress, resume_id, append_checkpoints)

    def _run_unlocked(self, progress, resume_id: str | None, append_checkpoints: bool) -> str:
        experiment_id = resume_id or str(uuid.uuid4())
        self.id = experiment_id
        if resume_id:
            existing = self.repository.get(experiment_id)
            saved_config = ExperimentConfig.model_validate(existing["config"])
            changed = saved_config != self.config
            if changed:
                previous = saved_config.model_dump(mode="json")
                current = self.config.model_dump(mode="json")
                previous_checkpoints = previous.pop("checkpoints")
                current_checkpoints = current.pop("checkpoints")
                valid_append = (
                    append_checkpoints
                    and previous == current
                    and current_checkpoints[: len(previous_checkpoints)] == previous_checkpoints
                    and len(current_checkpoints) > len(previous_checkpoints)
                )
                if not valid_append:
                    raise ConfigurationError(
                        "Resume configuration differs from the saved experiment"
                    )
            elif existing["status"] == "complete":
                raise ConfigurationError("Experiment is already complete")
            environment = existing["environment"]
            if self.dataset_hash != environment.get("dataset_sha256"):
                raise ConfigurationError("Dataset contents changed since the original run")
            if changed:
                self.repository.update_config(experiment_id, self.config)
            if changed:
                environment.setdefault("config_history_sha256", []).append(
                    environment.get("config_sha256")
                )
                environment["config_sha256"] = hashlib.sha256(
                    self.config.model_dump_json().encode()
                ).hexdigest()
                environment["checkpoint_config_sha256"] = environment_record(
                    self.config, Path.cwd()
                )["checkpoint_config_sha256"]
            completed = self.repository.completed_checkpoints(
                experiment_id, len(self.config.probes)
            )
            self.repository.resume(experiment_id)
        else:
            environment = environment_record(self.config, Path.cwd())
            if self.dataset_hash:
                environment["dataset_sha256"] = self.dataset_hash
            self.repository.create(experiment_id, self.config, environment)
            completed = set()
        started = time.perf_counter()
        model_load_seconds: dict[str, float] = {}
        runtime = None
        loaded_model: str | None = None
        try:
            if self.config.dataset:
                archive = self.project_dir / "experiments" / experiment_id
                archive.mkdir(mode=0o700, parents=True, exist_ok=True)
                snapshot = archive / "dataset.yaml"
                raw_dataset = Path(self.config.dataset).read_bytes()
                if hashlib.sha256(raw_dataset).hexdigest() != self.dataset_hash:
                    raise ConfigurationError(
                        "Dataset changed after validation; rerun with a stable file"
                    )
                if not snapshot.exists():
                    snapshot.write_bytes(raw_dataset)
                    snapshot.chmod(0o600)
                elif hashlib.sha256(snapshot.read_bytes()).hexdigest() != self.dataset_hash:
                    raise ConfigurationError(
                        "Archived dataset snapshot differs from the run dataset"
                    )
            for checkpoint in self.config.checkpoints:
                if checkpoint.id in completed:
                    if progress:
                        progress(f"checkpoint {checkpoint.id}: already complete, skipped")
                    continue
                model = checkpoint.model or self.config.target.model
                if self.config.max_model_size_gb and self.config.target.runtime in {
                    "mlx",
                    "transformers",
                    "llama_cpp",
                }:
                    model_path = Path(model).expanduser()
                    if model_path.is_file() and self.config.target.runtime == "llama_cpp":
                        weight_bytes = model_path.stat().st_size
                    elif model_path.is_dir():
                        weight_bytes = sum(
                            file.stat().st_size
                            for pattern in ("*.safetensors", "*.bin")
                            for file in model_path.glob(pattern)
                        )
                    else:
                        weight_bytes = 0
                    if weight_bytes > self.config.max_model_size_gb * 1024**3:
                        raise ConfigurationError(
                            f"Checkpoint {checkpoint.id} has {weight_bytes / 1024**3:.2f} GiB of weights, above max_model_size_gb={self.config.max_model_size_gb}"
                        )
                if runtime is None or model != loaded_model:
                    if runtime is not None:
                        runtime.unload()
                    runtime = create_runtime(
                        ModelSpec(
                            runtime=self.config.target.runtime,
                            model=model,
                            endpoint=self.config.target.endpoint,
                            device=self.config.target.device,
                        )
                    )
                    load_started = time.perf_counter()
                    runtime.load()
                    model_load_seconds[checkpoint.id] = round(time.perf_counter() - load_started, 4)
                    loaded_model = model
                    if progress:
                        progress(
                            f"model loaded: {model} ({model_load_seconds[checkpoint.id]:.2f}s)"
                        )
                log.info(
                    "Evaluating checkpoint %s with %s", checkpoint.id, self.config.target.runtime
                )
                if progress:
                    progress(f"checkpoint {checkpoint.id}: {model}")
                checkpoint_evidence: list[Evidence] = []
                checkpoint_observations: list[Observation] = []
                for probe_name in self.config.probes:
                    probe = get_probe(probe_name)
                    probe_evidence = []
                    cases = (
                        [
                            case
                            for case in self.dataset.cases
                            if case.probe == probe_name and case.split == "evaluation"
                        ]
                        if self.dataset
                        else probe.cases()
                    )
                    scores_by_case: dict[str, list[float]] = {}
                    for base_seed in self.config.effective_seeds:
                        for index, case in enumerate(cases):
                            actual_seed = base_seed + index
                            request = GenerationRequest(
                                system=case.system,
                                prompt=case.prompt,
                                seed=actual_seed,
                                max_new_tokens=self.config.max_new_tokens,
                                temperature=self.config.temperature,
                                probe=probe_name,
                                case_id=case.id,
                                checkpoint=checkpoint.id,
                            )
                            try:
                                response = runtime.generate(request)
                                if self.dataset:
                                    score, rationale, evaluation = evaluate_rubric(
                                        case, response.text
                                    )
                                else:
                                    score, rationale = probe.evaluate(case, response.text)
                                    evaluation = {}
                            except TraceAIError:
                                raise
                            except Exception as exc:
                                raise ProbeExecutionError(
                                    f"Probe {probe_name}, case {case.id}, checkpoint {checkpoint.id} failed: {exc}"
                                ) from exc
                            scores_by_case.setdefault(case.id, []).append(score)
                            evidence_case_id = (
                                f"{case.id}~s{actual_seed}"
                                if len(self.config.effective_seeds) > 1
                                else case.id
                            )
                            probe_evidence.append(
                                Evidence(
                                    experiment_id=experiment_id,
                                    checkpoint=checkpoint.id,
                                    probe=probe_name,
                                    case_id=evidence_case_id,
                                    base_case_id=case.id,
                                    prompt=case.prompt,
                                    system=case.system,
                                    output=response.text,
                                    score=score,
                                    rationale=rationale,
                                    evaluator=f"dataset:{self.dataset_hash[:12]}"
                                    if self.dataset_hash
                                    else f"{probe_name}:{probe.version}",
                                    seed=request.seed,
                                    model=model,
                                    runtime=self.config.target.runtime,
                                    duration_seconds=response.duration_seconds,
                                    token_count=response.token_count,
                                    evaluation=evaluation,
                                )
                            )
                    failures = sum(item.score for item in probe_evidence)
                    count = len(probe_evidence)
                    if self.config.schema_version == 2:
                        low, high = clustered_bootstrap_interval(scores_by_case, self.config.seed)
                        uncertainty_method = "case_cluster_bootstrap_1000"
                    else:
                        low, high = wilson_interval(failures, count)
                        uncertainty_method = "wilson_independent_cases"
                    checkpoint_observations.append(
                        Observation(
                            experiment_id=experiment_id,
                            checkpoint=checkpoint.id,
                            probe=probe_name,
                            score=failures / count,
                            sample_count=count,
                            ci_low=low,
                            ci_high=high,
                            evaluator=f"dataset:{self.dataset_hash[:12]}"
                            if self.dataset_hash
                            else f"{probe_name}:{probe.version}",
                            uncertainty_method=uncertainty_method,
                        )
                    )
                    checkpoint_evidence.extend(probe_evidence)
                    if progress:
                        progress(
                            f"  {probe_name}: {failures / count:.2f} failure rate from {count} cases"
                        )
                self.repository.add_checkpoint(checkpoint_observations, checkpoint_evidence)
                if progress:
                    progress(f"checkpoint complete: {checkpoint.id}")
            prior_execution = environment.get("execution", {})
            all_evidence = self.repository.get(experiment_id)["evidence"]
            total_inference = sum(item.duration_seconds for item in all_evidence)
            total_tokens = sum(item.token_count or 0 for item in all_evidence)
            environment["execution"] = {
                "duration_seconds": round(
                    prior_execution.get("duration_seconds", 0) + time.perf_counter() - started, 4
                ),
                "model_load_seconds_by_checkpoint": {
                    **prior_execution.get("model_load_seconds_by_checkpoint", {}),
                    **model_load_seconds,
                },
                "inference_seconds": round(total_inference, 4),
                "generations": len(all_evidence),
                "generated_tokens_when_reported": total_tokens,
                "reported_token_throughput": round(total_tokens / total_inference, 2)
                if total_tokens and total_inference
                else None,
            }
            self.repository.update_environment(experiment_id, environment)
            self.repository.finish(experiment_id, "complete")
        except KeyboardInterrupt:
            self.repository.finish(experiment_id, "failed", "interrupted by user")
            raise
        except Exception as exc:
            self.repository.finish(experiment_id, "failed", str(exc))
            raise
        finally:
            if runtime is not None:
                runtime.unload()
        return experiment_id

    def results(self, experiment_id: str | None = None) -> list[Observation]:
        return self.repository.get(experiment_id or self._require_id())["observations"]

    def report(self, experiment_id: str | None = None) -> Report:
        record = self.repository.get(experiment_id or self._require_id())
        config = ExperimentConfig.model_validate(record["config"])
        changes = self.analyzer.detect_changes(
            record["observations"], [item.id for item in config.checkpoints]
        )
        limitations = [
            "Failure rates reflect this fixed case set; they do not establish hidden goals, deception, or general model safety.",
            "The target is scored with explicit rules; monitor and agent fields are recorded but do not affect scores.",
        ]
        if config.schema_version == 2:
            limitations.append(
                "Case-cluster bootstrap intervals describe variation across these cases and seeds; they do not establish population generalization or causal change."
            )
        else:
            limitations.append(
                "Wilson intervals assume independent cases; related prompts can be correlated. Interval separation is descriptive, not a significance test."
            )
        if config.target.runtime == "mock":
            limitations.insert(
                0,
                "All outputs are synthetic fixtures; this report is a pipeline demonstration, not a model finding.",
            )
        if record["status"] != "complete":
            limitations.insert(
                0, "This run is incomplete; measurements may cover only earlier checkpoints."
            )
        return Report(
            experiment_id=record["id"],
            status=record["status"],
            config=config,
            environment=record["environment"],
            observations=record["observations"],
            changes=changes,
            evidence=record["evidence"],
            limitations=limitations,
        )

    def _require_id(self) -> str:
        if self.id is None:
            raise ConfigurationError("Run the experiment first or provide an experiment ID")
        return self.id
