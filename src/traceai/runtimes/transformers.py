"""Local-only Hugging Face text-generation adapter."""

import gc
import time
from pathlib import Path

from traceai.errors import ModelNotFoundError, RuntimeUnavailableError, UnsupportedCapabilityError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import Capability, GenerationRequest, GenerationResult, InspectionResult


class TransformersRuntime(ModelRuntime):
    def capabilities(self) -> set[Capability]:
        return {Capability.OUTPUTS, Capability.LOGITS, Capability.HIDDEN_STATES}

    def load(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Install the optional Transformers runtime: pip install 'traceai-local[transformers]'"
            ) from exc
        model_id = self.spec.model
        if Path(model_id).exists():
            model_id = str(Path(model_id).resolve())
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=True, trust_remote_code=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=False
            )
        except (OSError, ValueError) as exc:
            raise ModelNotFoundError(
                f"Cannot load local Transformers model {model_id!r}. Cache it first or supply a local directory. Details: {exc}"
            ) from exc
        self.torch = torch
        device = self.spec.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeUnavailableError("CUDA was selected but PyTorch reports no CUDA device")
        try:
            self.model.to(device)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeUnavailableError(
                f"Cannot move Transformers model to {device}: {exc}"
            ) from exc
        self.device = device
        self.model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt},
        ]
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"System: {request.system}\nUser: {request.prompt}\nAssistant:"
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        self.torch.manual_seed(request.seed)
        with self.torch.inference_mode():
            result = self.model.generate(
                **encoded,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0,
                temperature=request.temperature if request.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = result[0][encoded["input_ids"].shape[-1] :]
        return GenerationResult(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            duration_seconds=time.perf_counter() - start,
            token_count=len(new_tokens),
        )

    def inspect(
        self, capability: Capability, request: GenerationRequest | None = None
    ) -> InspectionResult:
        if capability not in {Capability.LOGITS, Capability.HIDDEN_STATES}:
            raise UnsupportedCapabilityError(
                f"Transformers inspection does not expose {capability.value}"
            )
        if request is None:
            raise UnsupportedCapabilityError(
                "A prompt is required for logits or hidden-state inspection"
            )
        if getattr(self, "model", None) is None or getattr(self, "tokenizer", None) is None:
            self.load()
        encoded = self.tokenizer(
            f"{request.system}\n{request.prompt}",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model(
                **encoded,
                output_hidden_states=capability == Capability.HIDDEN_STATES,
                return_dict=True,
            )
        if capability == Capability.LOGITS:
            values, indices = self.torch.topk(output.logits[0, -1].float(), k=5)
            measurements = {
                "input_tokens": int(encoded["input_ids"].shape[-1]),
                "last_token_top5": [
                    {
                        "token_id": int(token),
                        "token": self.tokenizer.decode([int(token)]),
                        "logit": round(float(value), 6),
                    }
                    for value, token in zip(values.tolist(), indices.tolist(), strict=True)
                ],
            }
            limitations = ["Top logits are for the next token only; logits are not probabilities."]
        else:
            if not output.hidden_states:
                raise UnsupportedCapabilityError("Model did not return hidden states")
            measurements = {
                "input_tokens": int(encoded["input_ids"].shape[-1]),
                "layers": [
                    {
                        "index": index,
                        "last_token_l2_norm": round(
                            float(self.torch.linalg.vector_norm(layer[0, -1].float())), 6
                        ),
                    }
                    for index, layer in enumerate(output.hidden_states)
                ],
            }
            limitations = [
                "Layer norms summarize a single prompt's final-token representation; no interpretation is implied."
            ]
        return InspectionResult(
            capability=capability,
            runtime="transformers",
            model=self.spec.model,
            measurements=measurements,
            limitations=limitations,
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
