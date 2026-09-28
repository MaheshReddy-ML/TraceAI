"""Create a runnable local-model study without downloading or loading weights."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from traceai.dataset import load_dataset
from traceai.errors import ConfigurationError
from traceai.probes import get_probe
from traceai.schemas import CheckpointSpec, ExperimentConfig, ModelSpec

LOCAL_RUNTIMES = {"transformers", "mlx", "llama_cpp"}
DEFAULT_PROBES = ["baseline", "reward_hacking", "specification_gaming"]


def _model_reference(runtime: str, value: str) -> str:
    if not value.strip():
        raise ConfigurationError("A model path or installed Ollama tag is required")
    if runtime == "ollama":
        return value.strip()
    path = Path(value).expanduser().resolve()
    if runtime in {"transformers", "mlx"}:
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ConfigurationError(
                f"{runtime} model {path} needs a local directory containing config.json"
            )
    elif runtime == "llama_cpp":
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ConfigurationError(f"GGUF model {path} needs an existing .gguf file")
    else:
        raise ConfigurationError(f"Choose transformers, mlx, llama_cpp, or ollama; got {runtime!r}")
    return str(path)


def build_model_study(
    *,
    runtime: str,
    model: str,
    checkpoint_entries: list[str],
    dataset_path: Path | None,
    probes: list[str] | None,
    seed: int,
    device: str,
    output: Path,
) -> tuple[ExperimentConfig, str]:
    """Return validated config and YAML; caller owns the refusal-to-overwrite write."""
    if runtime not in LOCAL_RUNTIMES | {"ollama"}:
        raise ConfigurationError("Choose transformers, mlx, llama_cpp, or ollama")
    if device != "auto" and runtime != "transformers":
        raise ConfigurationError("--device cpu/cuda is available only for transformers")
    target_model = _model_reference(runtime, model)
    checkpoints = []
    for entry in checkpoint_entries:
        checkpoint_id, separator, checkpoint_model = entry.partition("=")
        if not separator or not checkpoint_id.strip() or not checkpoint_model.strip():
            raise ConfigurationError("Each --checkpoint must be ID=LOCAL_PATH or ID=OLLAMA_TAG")
        try:
            checkpoints.append(
                CheckpointSpec(
                    id=checkpoint_id.strip(), model=_model_reference(runtime, checkpoint_model)
                )
            )
        except ValidationError as exc:
            raise ConfigurationError(f"Invalid checkpoint {checkpoint_id!r}: {exc}") from exc
    if len(checkpoints) > 1 and len({item.model for item in checkpoints}) != len(checkpoints):
        raise ConfigurationError(
            "Compared checkpoints need different model paths or Ollama tags; "
            "the runtime reuses a model when the path is unchanged"
        )
    if not checkpoints:
        checkpoints = [CheckpointSpec(id="baseline")]

    dataset = None
    if dataset_path is not None:
        source = dataset_path.expanduser().resolve()
        manifest, _ = load_dataset(source)
        available = list(
            dict.fromkeys(case.probe for case in manifest.cases if case.split == "evaluation")
        )
        selected = probes or available
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ConfigurationError(f"Dataset has no evaluation cases for: {', '.join(missing)}")
        output_parent = output.parent.resolve()
        dataset = (
            str(source.relative_to(output_parent))
            if source.is_relative_to(output_parent)
            else str(source)
        )
    else:
        selected = probes or DEFAULT_PROBES
    for name in selected:
        get_probe(name)

    try:
        config = ExperimentConfig(
            schema_version=2,
            name="local-model-study",
            description=(
                "User-supplied cases and deterministic rubrics; review calibration before inference."
                if dataset
                else "Selected output probes for an initial local-model walkthrough."
            ),
            target=ModelSpec(runtime=runtime, model=target_model, device=device),
            dataset=dataset,
            probes=selected,
            checkpoints=checkpoints,
            seed=seed,
            temperature=0,
        )
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid model study: {exc}") from exc

    payload = {
        "schema_version": config.schema_version,
        "name": config.name,
        "description": config.description,
        "target": {
            "runtime": config.target.runtime,
            "model": config.target.model,
            **({"device": config.target.device} if config.target.device != "auto" else {}),
        },
        **({"dataset": config.dataset} if config.dataset else {}),
        "probes": config.probes,
        "checkpoints": [
            {"id": item.id, **({"model": item.model} if item.model else {})}
            for item in config.checkpoints
        ],
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
    }
    return config, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
