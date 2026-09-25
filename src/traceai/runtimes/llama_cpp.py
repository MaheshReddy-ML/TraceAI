"""Optional GGUF text generation with llama-cpp-python."""

from __future__ import annotations

import time
from pathlib import Path

from traceai.errors import ModelNotFoundError, RuntimeUnavailableError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import GenerationRequest, GenerationResult


class LlamaCppRuntime(ModelRuntime):
    def load(self) -> None:
        path = Path(self.spec.model).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ModelNotFoundError(f"Expected a local .gguf model file: {path}")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Install GGUF support: pip install 'traceai-local[llama_cpp]'"
            ) from exc
        try:
            self.model = Llama(model_path=str(path), n_ctx=2048, verbose=False)
        except Exception as exc:
            raise ModelNotFoundError(f"Cannot load GGUF model {path}: {exc}") from exc

    def generate(self, request: GenerationRequest) -> GenerationResult:
        started = time.perf_counter()
        try:
            result = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": request.system},
                    {"role": "user", "content": request.prompt},
                ],
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                seed=request.seed,
            )
            content = result["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeUnavailableError(f"GGUF generation failed: {exc}") from exc
        return GenerationResult(
            text=content or "",
            duration_seconds=time.perf_counter() - started,
            token_count=result.get("usage", {}).get("completion_tokens"),
        )

    def unload(self) -> None:
        self.model = None
