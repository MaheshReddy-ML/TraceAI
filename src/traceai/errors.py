"""User-facing failures with stable CLI exit codes."""


class TraceAIError(Exception):
    exit_code = 1


class ConfigurationError(TraceAIError):
    exit_code = 2


class ModelNotFoundError(TraceAIError):
    exit_code = 3


class RuntimeUnavailableError(TraceAIError):
    exit_code = 4


class UnsupportedCapabilityError(TraceAIError):
    exit_code = 5


class ProbeExecutionError(TraceAIError):
    exit_code = 6


class StorageError(TraceAIError):
    exit_code = 7


class CheckpointLoadError(TraceAIError):
    exit_code = 8
