"""Runtime registration and model discovery."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlparse

from traceai.errors import ConfigurationError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import ModelSpec


def create_runtime(spec: ModelSpec) -> ModelRuntime:
    if spec.runtime == "mock":
        from traceai.runtimes.mock import MockRuntime

        return MockRuntime(spec)
    if spec.runtime == "transformers":
        from traceai.runtimes.transformers import TransformersRuntime

        return TransformersRuntime(spec)
    if spec.runtime == "mlx":
        from traceai.runtimes.mlx import MLXRuntime

        return MLXRuntime(spec)
    if spec.runtime == "ollama":
        from traceai.runtimes.ollama import OllamaRuntime

        return OllamaRuntime(spec)
    if spec.runtime == "training_state":
        from traceai.runtimes.training_state import TrainingStateRuntime

        return TrainingStateRuntime(spec)
    if spec.runtime == "llama_cpp":
        from traceai.runtimes.llama_cpp import LlamaCppRuntime

        return LlamaCppRuntime(spec)
    for plugin in entry_points(group="traceai.runtimes"):
        if plugin.name == spec.runtime:
            implementation = plugin.load()
            if not isinstance(implementation, type) or not issubclass(implementation, ModelRuntime):
                raise ConfigurationError(
                    f"Runtime plugin {spec.runtime!r} must subclass ModelRuntime"
                )
            return implementation(spec)
    raise ConfigurationError(f"Unsupported runtime: {spec.runtime}")


def discover_models() -> list[dict[str, str]]:
    """List local snapshots and Ollama tags without downloading anything."""
    found: list[dict[str, str]] = []
    cache = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    if cache.is_dir():
        for repo in sorted(cache.glob("models--*")):
            for snapshot in sorted((repo / "snapshots").glob("*")):
                config_path = snapshot / "config.json"
                if not snapshot.is_dir() or not config_path.exists():
                    continue
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not any(
                    name.endswith("ForCausalLM") for name in config.get("architectures", [])
                ):
                    continue
                if not list(snapshot.glob("*.safetensors")):
                    continue
                name = repo.name.removeprefix("models--").replace("--", "/")
                runtime = (
                    "mlx"
                    if "quantization" in config
                    or "-MLX-" in name.upper()
                    or name.startswith("mlx-community/")
                    else "transformers"
                )
                found.append(
                    {
                        "runtime": runtime,
                        "model": name,
                        "path": str(snapshot),
                        "size_bytes": sum(
                            file.stat().st_size for file in snapshot.glob("*.safetensors")
                        ),
                    }
                )
    gguf_dir = Path(os.environ.get("TRACEAI_MODEL_DIR", "")).expanduser()
    if os.environ.get("TRACEAI_MODEL_DIR") and gguf_dir.is_dir():
        for file in sorted(gguf_dir.glob("*.gguf")):
            found.append(
                {
                    "runtime": "llama_cpp",
                    "model": file.stem,
                    "path": str(file.resolve()),
                    "size_bytes": file.stat().st_size,
                }
            )
    endpoint = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        and not parsed.username
        and not parsed.password
    ):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{endpoint}/api/tags", timeout=1.5) as response:
                payload = json.load(response)
            for item in payload.get("models", []):
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    found.append(
                        {
                            "runtime": "ollama",
                            "model": item["name"],
                            "path": endpoint,
                            "size_bytes": item.get("size"),
                        }
                    )
        except (OSError, ValueError, urllib.error.URLError):
            pass
    return found


def runtime_status(models: list[dict] | None = None) -> dict[str, bool]:
    return {
        "mlx": find_spec("mlx_lm") is not None,
        "transformers": find_spec("transformers") is not None,
        "llama_cpp": find_spec("llama_cpp") is not None,
        "ollama": any(
            model["runtime"] == "ollama"
            for model in (models if models is not None else discover_models())
        ),
    }
