"""Local repository: SQLite is authoritative; JSONL is a portable evidence export."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from traceai.errors import StorageError
from traceai.schemas import Evidence, ExperimentConfig, Observation


class ExperimentRepository(Protocol):
    def create(self, experiment_id: str, config: ExperimentConfig, environment: dict) -> None: ...
    def add_checkpoint(self, observations: list[Observation], evidence: list[Evidence]) -> None: ...
    def update_environment(self, experiment_id: str, environment: dict) -> None: ...
    def finish(self, experiment_id: str, status: str, error: str | None = None) -> None: ...
    def get(self, experiment_id: str) -> dict: ...
    def completed_checkpoints(self, experiment_id: str, probe_count: int) -> set[str]: ...
    def resume(self, experiment_id: str) -> None: ...
    def update_config(self, experiment_id: str, config: ExperimentConfig) -> None: ...


@contextmanager
def project_run_lock(project_dir: Path):
    """Allow one model-heavy experiment per local project at a time."""
    project_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = project_dir / "run.lock"
    with path.open("a+b") as stream:
        os.chmod(path, 0o600)
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StorageError(f"Another experiment is already running in {project_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class SQLiteRepository:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir.resolve()
        self.project_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db_path = self.project_dir / "traceai.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version > 2:
                    raise StorageError(
                        f"Storage schema version {version} is newer than this TraceAI build"
                    )
                db.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                    config_json TEXT NOT NULL, environment_json TEXT NOT NULL,
                    status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS observations (
                    experiment_id TEXT NOT NULL, checkpoint TEXT NOT NULL, probe TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, checkpoint, probe),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    experiment_id TEXT NOT NULL, checkpoint TEXT NOT NULL, probe TEXT NOT NULL,
                    case_id TEXT NOT NULL, data_json TEXT NOT NULL,
                    PRIMARY KEY(experiment_id, checkpoint, probe, case_id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                """)
                if version < 2:
                    db.executescript("""
                    CREATE TABLE IF NOT EXISTS watch_sessions (
                        directory TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                        config_json TEXT NOT NULL, processed_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    CREATE TABLE IF NOT EXISTS jobs (
                        id TEXT PRIMARY KEY, config_json TEXT NOT NULL, status TEXT NOT NULL,
                        result_id TEXT, error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        claimed_at TEXT
                    );
                    """)
                    db.execute("PRAGMA user_version=2")
            os.chmod(self.db_path, 0o600)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(
                f"Cannot initialize project storage at {self.project_dir}: {exc}"
            ) from exc

    def create(self, experiment_id: str, config: ExperimentConfig, environment: dict) -> None:
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO experiments(id,schema_version,config_json,environment_json,status) VALUES(?,?,?,?,?)",
                    (
                        experiment_id,
                        config.schema_version,
                        config.model_dump_json(),
                        json.dumps(environment),
                        "running",
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Cannot create experiment: {exc}") from exc

    def add_checkpoint(self, observations: list[Observation], evidence: list[Evidence]) -> None:
        try:
            with self._connect() as db:
                db.executemany(
                    "INSERT INTO observations VALUES(?,?,?,?)",
                    [
                        (item.experiment_id, item.checkpoint, item.probe, item.model_dump_json())
                        for item in observations
                    ],
                )
                db.executemany(
                    "INSERT INTO evidence VALUES(?,?,?,?,?)",
                    [
                        (
                            item.experiment_id,
                            item.checkpoint,
                            item.probe,
                            item.case_id,
                            item.model_dump_json(),
                        )
                        for item in evidence
                    ],
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Cannot persist checkpoint results: {exc}") from exc

    def finish(self, experiment_id: str, status: str, error: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET status=?, error=? WHERE id=?",
                (status, error, experiment_id),
            )
        if status == "complete":
            self.export_evidence(experiment_id)

    def update_environment(self, experiment_id: str, environment: dict) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET environment_json=? WHERE id=?",
                (json.dumps(environment), experiment_id),
            )

    def resume(self, experiment_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET status='running', error=NULL WHERE id=?", (experiment_id,)
            )

    def update_config(self, experiment_id: str, config: ExperimentConfig) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE experiments SET config_json=?, schema_version=? WHERE id=?",
                (config.model_dump_json(), config.schema_version, experiment_id),
            )

    def completed_checkpoints(self, experiment_id: str, probe_count: int) -> set[str]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT checkpoint FROM observations WHERE experiment_id=? GROUP BY checkpoint HAVING COUNT(*)=?",
                (experiment_id, probe_count),
            ).fetchall()
        return {row[0] for row in rows}

    def get(self, experiment_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None:
                raise StorageError(f"Experiment {experiment_id!r} not found in {self.db_path}")
            if row["schema_version"] not in {1, 2}:
                raise StorageError(
                    f"Experiment {experiment_id!r} uses unsupported schema version {row['schema_version']}"
                )
            observations = db.execute(
                "SELECT data_json FROM observations WHERE experiment_id=? ORDER BY rowid",
                (experiment_id,),
            ).fetchall()
            evidence = db.execute(
                "SELECT data_json FROM evidence WHERE experiment_id=? ORDER BY rowid",
                (experiment_id,),
            ).fetchall()
        return {
            "id": row["id"],
            "status": row["status"],
            "error": row["error"],
            "config": json.loads(row["config_json"]),
            "environment": json.loads(row["environment_json"]),
            "observations": [Observation.model_validate_json(item[0]) for item in observations],
            "evidence": [Evidence.model_validate_json(item[0]) for item in evidence],
        }

    def list_experiments(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,status,created_at,config_json FROM experiments ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "name": json.loads(row["config_json"])["name"],
            }
            for row in rows
        ]

    def get_watch_session(self, directory: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT experiment_id,config_json,processed_json FROM watch_sessions WHERE directory=?",
                (directory,),
            ).fetchone()
        if row is None:
            return None
        return {
            "experiment_id": row["experiment_id"],
            "config": json.loads(row["config_json"]),
            "processed": json.loads(row["processed_json"]),
        }

    def save_watch_session(
        self, directory: str, experiment_id: str, config: ExperimentConfig, processed: list[dict]
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO watch_sessions(directory,experiment_id,config_json,processed_json) VALUES(?,?,?,?) "
                "ON CONFLICT(directory) DO UPDATE SET experiment_id=excluded.experiment_id,config_json=excluded.config_json,processed_json=excluded.processed_json,updated_at=CURRENT_TIMESTAMP",
                (directory, experiment_id, config.model_dump_json(), json.dumps(processed)),
            )

    def submit_job(self, config: ExperimentConfig) -> str:
        job_id = str(uuid.uuid4())
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs(id,config_json,status) VALUES(?,?,'queued')",
                (job_id, config.model_dump_json()),
            )
        return job_id

    def claim_job(self) -> dict | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,config_json FROM jobs WHERE status='queued' ORDER BY created_at,rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='running',claimed_at=CURRENT_TIMESTAMP WHERE id=?",
                (row["id"],),
            )
        return {"id": row["id"], "config": json.loads(row["config_json"])}

    def finish_job(
        self, job_id: str, *, result_id: str | None = None, error: str | None = None
    ) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status=?,result_id=?,error=? WHERE id=? AND status='running'",
                ("failed" if error else "complete", result_id, error, job_id),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"Job {job_id!r} is not running")

    def list_jobs(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,status,result_id,error,created_at,claimed_at FROM jobs ORDER BY created_at DESC,rowid DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise StorageError(f"Job {job_id!r} was not found")
        return {**dict(row), "config": json.loads(row["config_json"])}

    def retry_job(self, job_id: str) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status='queued',result_id=NULL,error=NULL,claimed_at=NULL "
                "WHERE id=? AND status IN ('failed','running')",
                (job_id,),
            )
            if cursor.rowcount != 1:
                raise StorageError("Only failed or abandoned running jobs can be retried")

    def import_record(self, record: dict) -> None:
        """Persist a completed worker result with the same schema validation as local runs."""
        config = ExperimentConfig.model_validate(record["config"])
        observations = [Observation.model_validate(item) for item in record["observations"]]
        evidence = [Evidence.model_validate(item) for item in record["evidence"]]
        experiment_id = str(record["id"])
        try:
            if str(uuid.UUID(experiment_id)) != experiment_id:
                raise ValueError("noncanonical UUID")
        except ValueError as exc:
            raise StorageError("Worker result experiment ID must be a UUID") from exc
        if record["status"] != "complete":
            raise StorageError("Only complete worker results can be imported")
        if any(item.experiment_id != experiment_id for item in observations + evidence):
            raise StorageError("Worker result contains mismatched experiment IDs")
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO experiments(id,schema_version,config_json,environment_json,status) "
                    "VALUES(?,?,?,?,?)",
                    (
                        experiment_id,
                        config.schema_version,
                        config.model_dump_json(),
                        json.dumps(record["environment"]),
                        "complete",
                    ),
                )
                db.executemany(
                    "INSERT INTO observations VALUES(?,?,?,?)",
                    [
                        (item.experiment_id, item.checkpoint, item.probe, item.model_dump_json())
                        for item in observations
                    ],
                )
                db.executemany(
                    "INSERT INTO evidence VALUES(?,?,?,?,?)",
                    [
                        (
                            item.experiment_id,
                            item.checkpoint,
                            item.probe,
                            item.case_id,
                            item.model_dump_json(),
                        )
                        for item in evidence
                    ],
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Cannot import worker result: {exc}") from exc
        self.export_evidence(experiment_id)

    def export_evidence(self, experiment_id: str) -> Path:
        record = self.get(experiment_id)
        target_dir = self.project_dir / "experiments" / experiment_id
        target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = target_dir / "evidence.jsonl"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for item in record["evidence"]:
                stream.write(item.model_dump_json() + "\n")
        os.chmod(temporary, 0o600)
        temporary.replace(target)
        return target
