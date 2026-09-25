"""Authenticated coordinator and remote worker for independent experiments.

Workers execute one whole experiment per job. Checkpoint parallelism is not claimed.
Workers are trusted processes and must opt into a coordinator with a shared secret.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import ssl
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from traceai.engine import Experiment
from traceai.errors import ConfigurationError, StorageError
from traceai.schemas import ExperimentConfig
from traceai.storage import SQLiteRepository

MAX_BODY = 16 * 1024 * 1024


def _token() -> str:
    token = os.environ.get("TRACEAI_WORKER_TOKEN", "")
    if len(token) < 24:
        raise ConfigurationError("Set TRACEAI_WORKER_TOKEN to at least 24 random characters")
    return token


def serve_coordinator(
    repository: SQLiteRepository,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    cert: Path | None = None,
    key: Path | None = None,
) -> None:
    token = _token()
    if not 1 <= port <= 65535:
        raise ConfigurationError("Coordinator port must be between 1 and 65535")
    if host not in {"127.0.0.1", "localhost", "::1"} and not (cert and key):
        raise ConfigurationError("A TLS certificate and key are required for non-loopback workers")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            authorization = self.headers.get("Authorization", "")
            if not hmac.compare_digest(authorization, f"Bearer {token}"):
                self.send_error(401)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                self.send_error(413)
                return
            try:
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/v1/jobs/claim":
                    result = {"job": repository.claim_job()}
                elif self.path == "/v1/jobs/result":
                    job_id = str(body["job_id"])
                    record = body["record"]
                    job = repository.get_job(job_id)
                    if job["status"] != "running" or job["config"] != record["config"]:
                        raise StorageError("Worker result does not match a running job")
                    snapshot = None
                    if "dataset_snapshot_b64" in record:
                        try:
                            snapshot = base64.b64decode(
                                record["dataset_snapshot_b64"], validate=True
                            )
                        except (ValueError, binascii.Error) as exc:
                            raise StorageError("Invalid worker dataset snapshot") from exc
                        if hashlib.sha256(snapshot).hexdigest() != record["environment"].get(
                            "dataset_sha256"
                        ):
                            raise StorageError(
                                "Worker dataset snapshot hash differs from provenance"
                            )
                    try:
                        saved = repository.get(record["id"])
                    except StorageError:
                        saved = None
                    if saved is None:
                        repository.import_record(record)
                    elif saved["status"] != "complete" or saved["config"] != record["config"]:
                        raise StorageError("Conflicting experiment result already exists")
                    if snapshot is not None:
                        target = (
                            repository.project_dir / "experiments" / record["id"] / "dataset.yaml"
                        )
                        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        target.write_bytes(snapshot)
                        target.chmod(0o600)
                    repository.finish_job(job_id, result_id=record["id"])
                    result = {"accepted": True, "experiment_id": record["id"]}
                elif self.path == "/v1/jobs/failure":
                    repository.finish_job(str(body["job_id"]), error=str(body["error"])[:2000])
                    result = {"accepted": True}
                else:
                    self.send_error(404)
                    return
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                StorageError,
                ConfigurationError,
            ) as exc:
                self._json({"error": str(exc)}, 400)
                return
            self._json(result)

        def _json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise ConfigurationError(f"Cannot bind worker coordinator: {exc}") from exc
    if cert and key:
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(cert), str(key))
            server.socket = context.wrap_socket(server.socket, server_side=True)
        except (OSError, ssl.SSLError) as exc:
            server.server_close()
            raise ConfigurationError(f"Cannot configure coordinator TLS: {exc}") from exc
    scheme = "https" if cert else "http"
    print(f"TraceAI worker coordinator: {scheme}://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class RemoteWorker:
    def __init__(
        self,
        server: str,
        workdir: Path,
        *,
        model_root: Path,
        ca_file: Path | None = None,
        max_model_size_gb: float = 16.0,
    ):
        parsed = urlparse(server)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ConfigurationError(
                "Worker server must be an HTTP(S) URL without embedded credentials"
            )
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigurationError("Remote worker traffic requires HTTPS")
        self.server = server.rstrip("/")
        self.token = _token()
        self.workdir = workdir.resolve()
        self.model_root = model_root.resolve()
        if max_model_size_gb <= 0:
            raise ConfigurationError("Worker maximum model size must be positive")
        self.max_model_size_gb = max_model_size_gb
        try:
            self.context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
        except (OSError, ssl.SSLError) as exc:
            raise ConfigurationError(f"Cannot load worker CA file: {exc}") from exc

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.server + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30, context=self.context) as response:
                return json.load(response)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            raise StorageError(f"Worker coordinator request failed: {exc}") from exc

    def run_once(self) -> str | None:
        job = self._post("/v1/jobs/claim", {})["job"]
        if job is None:
            return None
        job_id = job["id"]
        try:
            config = ExperimentConfig.model_validate(job["config"])
            self._validate_paths(config)
            project = self.workdir / job_id
            repository = SQLiteRepository(project)
            experiment = Experiment(config, project, repository)
            result_id = experiment.run()
            record = repository.get(result_id)
            payload = {
                **record,
                "observations": [item.model_dump(mode="json") for item in record["observations"]],
                "evidence": [item.model_dump(mode="json") for item in record["evidence"]],
            }
            snapshot = project / "experiments" / result_id / "dataset.yaml"
            if snapshot.is_file():
                payload["dataset_snapshot_b64"] = base64.b64encode(snapshot.read_bytes()).decode(
                    "ascii"
                )
            self._post("/v1/jobs/result", {"job_id": job_id, "record": payload})
            return result_id
        except Exception as exc:
            try:
                self._post("/v1/jobs/failure", {"job_id": job_id, "error": str(exc)})
            except StorageError:
                pass
            raise

    def _validate_paths(self, config: ExperimentConfig) -> None:
        if config.dataset:
            path = Path(config.dataset).expanduser().resolve()
            if not path.is_file() or not path.is_relative_to(self.model_root):
                raise ConfigurationError("Worker dataset must be a file inside --model-root")
        if config.target.runtime == "mock":
            return
        if config.target.runtime not in {"mlx", "transformers", "llama_cpp"}:
            raise ConfigurationError(
                "Workers allow only mock, MLX, Transformers, and GGUF runtimes"
            )
        for checkpoint in config.checkpoints:
            path = Path(checkpoint.model or config.target.model).expanduser().resolve()
            valid = (
                path.is_file() and path.suffix.lower() == ".gguf"
                if config.target.runtime == "llama_cpp"
                else path.is_dir()
            )
            if not valid or not path.is_relative_to(self.model_root):
                raise ConfigurationError("Worker model checkpoints must be inside --model-root")
            weight_bytes = (
                path.stat().st_size
                if path.is_file()
                else sum(
                    file.stat().st_size
                    for pattern in ("*.safetensors", "*.bin")
                    for file in path.glob(pattern)
                )
            )
            if weight_bytes > self.max_model_size_gb * 1024**3:
                raise ConfigurationError("Worker checkpoint exceeds --max-model-size-gb")
