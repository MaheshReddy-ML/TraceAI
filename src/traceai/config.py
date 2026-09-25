"""Validated, versioned YAML configuration."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from traceai.errors import ConfigurationError
from traceai.schemas import ExperimentConfig

DEFAULT_EXPERIMENT = """schema_version: 1
name: local-behavior-study
description: Synthetic walkthrough; replace target with a local runtime for real observations.
target:
  runtime: mock
  model: deterministic-fixture
probes:
  - baseline
  - specification_gaming
  - reward_hacking
  - sycophancy
  - leakage
checkpoints:
  - id: '1'
  - id: '5'
  - id: '10'
  - id: '15'
seed: 42
max_new_tokens: 96
temperature: 0
"""


def load_config(path: Path) -> ExperimentConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = ExperimentConfig.model_validate(payload)
        if config.dataset:
            config = config.model_copy(
                update={"dataset": str((path.parent / config.dataset).resolve())}
            )
        return config
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise ConfigurationError(f"Invalid experiment file {path}: {exc}") from exc
