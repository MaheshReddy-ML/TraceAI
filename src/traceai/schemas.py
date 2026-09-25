"""Versioned experiment and evidence schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Capability(StrEnum):
    OUTPUTS = "outputs"
    LOGITS = "logits"
    HIDDEN_STATES = "hidden_states"
    ATTENTION = "attention"
    GRADIENTS = "gradients"
    TRAINING_STATE = "training_state"
    CHECKPOINTS = "checkpoints"


class ModelSpec(StrictModel):
    runtime: str
    model: str = Field(min_length=1)
    endpoint: str | None = None
    device: str = "auto"

    @field_validator("runtime")
    @classmethod
    def supported_runtime(cls, value: str) -> str:
        if not value or not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("runtime must be a simple plugin name")
        return value

    @field_validator("device")
    @classmethod
    def supported_device(cls, value: str) -> str:
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        return value

    @model_validator(mode="after")
    def device_matches_runtime(self) -> ModelSpec:
        if self.device != "auto" and self.runtime != "transformers":
            raise ValueError("explicit device selection is supported only for transformers")
        return self


class CheckpointSpec(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    model: str | None = None


class ExperimentConfig(StrictModel):
    schema_version: int = 1
    name: str = Field(min_length=1, max_length=150)
    description: str = ""
    target: ModelSpec
    monitor: ModelSpec | None = None
    agent: ModelSpec | None = None
    probes: list[str] = Field(
        default_factory=lambda: [
            "baseline",
            "specification_gaming",
            "reward_hacking",
            "sycophancy",
            "leakage",
        ]
    )
    checkpoints: list[CheckpointSpec] = Field(
        default_factory=lambda: [CheckpointSpec(id="baseline")]
    )
    seed: int = 42
    seeds: list[int] | None = None
    max_new_tokens: int = Field(default=96, ge=8, le=2048)
    temperature: float = Field(default=0, ge=0, le=2)
    dataset: str | None = None
    max_model_size_gb: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_experiment(self) -> ExperimentConfig:
        if self.schema_version not in {1, 2}:
            raise ValueError("experiment schema_version must be 1 or 2")
        if self.target.runtime == "training_state":
            raise ValueError("training_state is inspection-only and cannot run behavior probes")
        if not self.probes or len(set(self.probes)) != len(self.probes):
            raise ValueError("probes must be a nonempty list without duplicates")
        ids = [checkpoint.id for checkpoint in self.checkpoints]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("checkpoints must be a nonempty list with unique IDs")
        if (
            self.target.runtime != "mock"
            and len(ids) > 1
            and any(not c.model for c in self.checkpoints)
        ):
            raise ValueError("each real-model checkpoint needs a model path or model name")
        if self.schema_version == 1 and (self.dataset is not None or self.seeds is not None):
            raise ValueError("datasets and repeated seeds require schema_version 2")
        if self.seeds is not None and (not self.seeds or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be a nonempty list without duplicates")
        return self

    @property
    def effective_seeds(self) -> list[int]:
        return self.seeds if self.seeds is not None else [self.seed]


class GenerationRequest(StrictModel):
    system: str
    prompt: str
    seed: int
    max_new_tokens: int
    temperature: float
    probe: str
    case_id: str
    checkpoint: str


class GenerationResult(StrictModel):
    text: str
    duration_seconds: float = Field(ge=0)
    token_count: int | None = None


class InspectionResult(StrictModel):
    capability: Capability
    runtime: str
    model: str
    measurements: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)


class Evidence(StrictModel):
    experiment_id: str
    checkpoint: str
    probe: str
    case_id: str
    base_case_id: str | None = None
    prompt: str
    system: str
    output: str
    score: float = Field(ge=0, le=1)
    rationale: str
    evaluator: str
    seed: int
    model: str
    runtime: str
    duration_seconds: float
    token_count: int | None = None
    evaluation: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Observation(StrictModel):
    experiment_id: str
    checkpoint: str
    probe: str
    score: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=1)
    ci_low: float = Field(ge=0, le=1)
    ci_high: float = Field(ge=0, le=1)
    evaluator: str
    uncertainty_method: str = "wilson_independent_cases"


class ChangePoint(StrictModel):
    probe: str
    from_checkpoint: str
    to_checkpoint: str
    delta: float
    classification: str
    method: str


class Report(StrictModel):
    schema_version: int = 2
    experiment_id: str
    status: str
    config: ExperimentConfig
    environment: dict[str, Any]
    observations: list[Observation]
    changes: list[ChangePoint]
    evidence: list[Evidence]
    limitations: list[str]
