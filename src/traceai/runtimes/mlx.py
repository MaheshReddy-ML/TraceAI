"""Local snapshot adapter for mlx-lm on Apple Silicon."""

import gc
import time
from pathlib import Path

from traceai.errors import ModelNotFoundError, RuntimeUnavailableError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import GenerationRequest, GenerationResult


class MLXRuntime(ModelRuntime):
    def load(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Install mlx-lm on Apple Silicon: pip install 'traceai-local[mlx]'"
            ) from exc
        path = Path(self.spec.model).expanduser()
        if not path.is_dir() or not (path / "config.json").exists():
            raise ModelNotFoundError(
                "MLX requires a local model snapshot directory containing config.json; use 'traceai models' to find one"
            )
        try:
            self.model, self.tokenizer = load(str(path.resolve()))
        except (OSError, ValueError) as exc:
            raise ModelNotFoundError(f"Cannot load MLX model at {path}: {exc}") from exc

    def generate(self, request: GenerationRequest) -> GenerationResult:
        try:
            import mlx.core as mx
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeUnavailableError("Installed mlx-lm is missing generation helpers") from exc
        mx.random.seed(request.seed)
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"System: {request.system}\nUser: {request.prompt}\nAssistant:"
        start = time.perf_counter()
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=request.max_new_tokens,
            sampler=make_sampler(temp=request.temperature),
            verbose=False,
        )
        return GenerationResult(text=text, duration_seconds=time.perf_counter() - start)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
