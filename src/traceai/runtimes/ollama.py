"""Local Ollama adapter; no cloud endpoints or model pulls."""

import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from traceai.errors import ModelNotFoundError, RuntimeUnavailableError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import GenerationRequest, GenerationResult


class OllamaRuntime(ModelRuntime):
    def __init__(self, spec):
        super().__init__(spec)
        self.endpoint = (spec.endpoint or "http://127.0.0.1:11434").rstrip("/")
        parsed = urlparse(self.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username
            or parsed.password
        ):
            raise RuntimeUnavailableError("Ollama endpoint must be a local HTTP address")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def load(self) -> None:
        try:
            with self.opener.open(f"{self.endpoint}/api/tags", timeout=3) as response:
                models = json.load(response).get("models", [])
        except (OSError, ValueError) as exc:
            raise RuntimeUnavailableError(
                f"Cannot reach local Ollama at {self.endpoint}: {exc}"
            ) from exc
        if self.spec.model not in {item.get("name") for item in models}:
            raise ModelNotFoundError(
                f"Ollama model {self.spec.model!r} is not installed. Run 'ollama list'; TraceAI will not pull it"
            )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = {
            "model": self.spec.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "options": {
                "seed": request.seed,
                "temperature": request.temperature,
                "num_predict": request.max_new_tokens,
            },
        }
        body = json.dumps(payload).encode()
        http_request = urllib.request.Request(
            f"{self.endpoint}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with self.opener.open(http_request, timeout=180) as response:
                result = json.load(response)
        except (OSError, ValueError) as exc:
            raise RuntimeUnavailableError(f"Ollama generation failed: {exc}") from exc
        return GenerationResult(
            text=result["message"]["content"],
            duration_seconds=time.perf_counter() - start,
            token_count=result.get("eval_count"),
        )

    def unload(self) -> None:
        pass
