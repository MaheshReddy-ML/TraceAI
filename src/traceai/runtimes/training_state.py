"""Safe, read-only inspection of Hugging Face Trainer checkpoint state."""

import json
import math
from pathlib import Path

from traceai.errors import CheckpointLoadError, UnsupportedCapabilityError
from traceai.runtimes.base import ModelRuntime
from traceai.schemas import Capability, GenerationRequest, GenerationResult, InspectionResult


def _reject_nonfinite(value: str):
    raise ValueError(f"non-finite value {value}")


class TrainingStateRuntime(ModelRuntime):
    def load(self) -> None:
        self.path = Path(self.spec.model).expanduser().resolve()
        state_file = self.path / "trainer_state.json"
        if not self.path.is_dir() or not state_file.is_file():
            raise CheckpointLoadError(f"Expected a local trainer_state.json in {self.path}")
        if state_file.stat().st_size > 10 * 1024 * 1024:
            raise CheckpointLoadError("trainer_state.json exceeds the 10 MiB inspection limit")
        try:
            state = json.loads(
                state_file.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite
            )
        except (OSError, ValueError) as exc:
            raise CheckpointLoadError(f"Cannot parse {state_file}: {exc}") from exc
        if not isinstance(state, dict):
            raise CheckpointLoadError("Trainer state must be a JSON object")
        self.state = state

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise UnsupportedCapabilityError(
            "training_state is inspection-only; select a generation runtime for probes"
        )

    def capabilities(self) -> set[Capability]:
        return {Capability.TRAINING_STATE, Capability.CHECKPOINTS}

    def inspect(
        self, capability: Capability, request: GenerationRequest | None = None
    ) -> InspectionResult:
        if capability != Capability.TRAINING_STATE:
            raise UnsupportedCapabilityError(
                f"training_state runtime does not expose {capability.value}"
            )
        if getattr(self, "state", None) is None:
            self.load()
        history = self.state.get("log_history", [])
        if not isinstance(history, list):
            history = []
        allowed_metrics = {
            "step",
            "epoch",
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "train_loss",
            "eval_accuracy",
        }
        measurements = {
            "global_step": self.state.get("global_step"),
            "epoch": self.state.get("epoch"),
            "best_metric": self.state.get("best_metric"),
            "log_history": [
                {
                    key: value
                    for key, value in row.items()
                    if key in allowed_metrics
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                }
                for row in history[-1000:]
                if isinstance(row, dict)
            ],
        }
        return InspectionResult(
            capability=capability,
            runtime="training_state",
            model=str(self.path),
            measurements=measurements,
            limitations=[
                "This reads saved Trainer metrics only; it does not observe live gradients or activations."
            ],
        )

    def unload(self) -> None:
        self.state = None
