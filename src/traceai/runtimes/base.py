"""Model runtime contract."""

from abc import ABC, abstractmethod

from traceai.errors import UnsupportedCapabilityError
from traceai.schemas import (
    Capability,
    GenerationRequest,
    GenerationResult,
    InspectionResult,
    ModelSpec,
)


class ModelRuntime(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def inspect(
        self, capability: Capability, request: GenerationRequest | None = None
    ) -> InspectionResult:
        raise UnsupportedCapabilityError(
            f"{self.spec.runtime} does not expose {capability.value} through TraceAI 0.1"
        )

    def capabilities(self) -> set[Capability]:
        return {Capability.OUTPUTS}

    @abstractmethod
    def unload(self) -> None: ...
