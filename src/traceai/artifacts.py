"""Explicit export of completed experiment artifacts to S3-compatible storage."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from traceai.errors import ConfigurationError, RuntimeUnavailableError, StorageError
from traceai.storage import SQLiteRepository


def upload_experiment(
    repository: SQLiteRepository, experiment_id: str, destination: str, *, client=None
) -> dict:
    parsed = urlparse(destination)
    if (
        parsed.scheme != "s3"
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigurationError("Artifact destination must be s3://bucket/prefix")
    record = repository.get(experiment_id)
    if record["status"] != "complete":
        raise ConfigurationError("Only completed experiments can be exported")
    if client is None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeUnavailableError(
                "Install the S3 extra: pip install 'traceai-local[s3]'"
            ) from exc
        client = boto3.client("s3")
    prefix = parsed.path.strip("/")
    prefix = "/".join(part for part in (prefix, experiment_id) if part)
    artifact_dir = repository.project_dir / "experiments" / experiment_id
    evidence = artifact_dir / "evidence.jsonl"
    if not evidence.is_file():
        evidence = repository.export_evidence(experiment_id)
    files: list[tuple[str, bytes, str]] = [
        (
            "report.json",
            json.dumps(
                {
                    **record,
                    "observations": [
                        item.model_dump(mode="json") for item in record["observations"]
                    ],
                    "evidence": [item.model_dump(mode="json") for item in record["evidence"]],
                },
                ensure_ascii=False,
                default=str,
            ).encode(),
            "application/json",
        ),
        ("evidence.jsonl", evidence.read_bytes(), "application/x-ndjson"),
    ]
    dataset = artifact_dir / "dataset.yaml"
    if dataset.is_file():
        files.append(("dataset.yaml", dataset.read_bytes(), "application/yaml"))
    keys = []
    for filename, body, content_type in files:
        key = f"{prefix}/{filename}"
        try:
            client.put_object(
                Bucket=parsed.netloc,
                Key=key,
                Body=body,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )
        except Exception as exc:
            raise StorageError(f"S3 upload failed for {key}: {exc}") from exc
        keys.append(key)
    return {
        "bucket": parsed.netloc,
        "keys": keys,
        "note": "Explicit export only; SQLite remains the local source of truth.",
    }
