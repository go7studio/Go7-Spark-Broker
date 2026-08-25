from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .contract import canonical_json, request_digest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class IdempotencyConflict(RuntimeError):
    pass


class QueueFull(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection
        return connection

    def close(self) -> None:
        """Close the SQLite connection owned by the calling thread.

        Connections are thread-local because HTTP handlers and the scheduler
        access the store concurrently.  Each of those threads must release its
        own connection when it exits; closing a connection from another thread
        is not permitted by sqlite.
        """
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    def _initialize(self) -> None:
        connection = self._connection()
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                capability TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                profile_id TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                attempt INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                UNIQUE(origin, idempotency_key)
            );
            CREATE INDEX IF NOT EXISTS jobs_queue_idx
                ON jobs(status, priority DESC, created_at ASC);
            CREATE TABLE IF NOT EXISTS job_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                status TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS job_events_job_idx ON job_events(job_id, sequence);
            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
                kind TEXT NOT NULL,
                role TEXT NOT NULL,
                media_type TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS broker_epochs (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                epoch INTEGER NOT NULL,
                instance_id TEXT NOT NULL,
                started_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS resource_leases (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                broker_epoch INTEGER NOT NULL,
                fencing_token TEXT NOT NULL UNIQUE,
                resource_group TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                route_id TEXT,
                service_class TEXT NOT NULL,
                mode TEXT NOT NULL,
                estimated_memory_gb INTEGER NOT NULL,
                status TEXT NOT NULL,
                throttle_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                released_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS active_resource_group_idx
                ON resource_leases(resource_group)
                WHERE status IN ('acquiring','active','releasing','unknown');
            CREATE INDEX IF NOT EXISTS resource_leases_job_idx
                ON resource_leases(job_id, created_at);
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()}
        if "not_before" not in columns:
            connection.execute("ALTER TABLE jobs ADD COLUMN not_before TEXT")

    def begin_broker_epoch(self, instance_id: str, *, minimum_epoch: int = 1) -> int:
        if minimum_epoch < 1:
            raise ValueError("minimum epoch must be positive")
        now = utc_now()
        with self.transaction(immediate=True) as tx:
            row = tx.execute("SELECT epoch FROM broker_epochs WHERE singleton=1").fetchone()
            epoch = max(int(row["epoch"]) + 1 if row else 1, minimum_epoch)
            tx.execute(
                """INSERT INTO broker_epochs(singleton,epoch,instance_id,started_at) VALUES(1,?,?,?)
                   ON CONFLICT(singleton) DO UPDATE SET epoch=excluded.epoch,instance_id=excluded.instance_id,started_at=excluded.started_at""",
                (epoch, instance_id, now),
            )
        return epoch

    def interrupt_uncertain_jobs(self) -> list[str]:
        """Mark active jobs interrupted after runtime reconciliation has run."""
        now = utc_now()
        interrupted: list[str] = []
        with self.transaction(immediate=True) as tx:
            rows = tx.execute("SELECT id FROM jobs WHERE status IN ('loading','running','validating')").fetchall()
            for row in rows:
                interrupted.append(row["id"])
                error = {
                    "code": "broker_restarted",
                    "message": "broker restarted while job was active; runtime state was not safely reattached",
                    "retryable": False,
                }
                tx.execute(
                    "UPDATE jobs SET status='interrupted', updated_at=?, finished_at=?, error_json=? WHERE id=?",
                    (now, now, canonical_json(error), row["id"]),
                )
                tx.execute(
                    "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                    (row["id"], "interrupted", canonical_json({"reason": "broker_restart"}), now),
                )
        return interrupted

    def create_resource_lease(
        self,
        *,
        job_id: str,
        broker_epoch: int,
        resource_group: str,
        profile_id: str,
        route_id: str | None,
        service_class: str,
        mode: str,
        estimated_memory_gb: int,
        throttle: list[dict[str, object]],
        decision: dict[str, object],
    ) -> dict[str, Any]:
        now = utc_now()
        lease_id = f"lease_{uuid.uuid4().hex}"
        fencing_token = f"fence_{broker_epoch}_{uuid.uuid4().hex}"
        with self.transaction(immediate=True) as tx:
            tx.execute(
                """INSERT INTO resource_leases(
                    id,job_id,broker_epoch,fencing_token,resource_group,profile_id,route_id,
                    service_class,mode,estimated_memory_gb,status,throttle_json,decision_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    lease_id, job_id, broker_epoch, fencing_token, resource_group, profile_id,
                    route_id, service_class, mode, estimated_memory_gb, "acquiring",
                    canonical_json(throttle), canonical_json(decision), now,
                ),
            )
        return self.get_resource_lease(lease_id)  # type: ignore[return-value]

    def set_resource_lease_status(
        self,
        lease_id: str,
        status: str,
        *,
        throttle: list[dict[str, object]] | None = None,
        decision: dict[str, object] | None = None,
    ) -> None:
        if status not in {"acquiring", "active", "releasing", "released", "unknown", "denied"}:
            raise ValueError("invalid resource lease status")
        now = utc_now()
        released_at = now if status in {"released", "denied"} else None
        with self.transaction(immediate=True) as tx:
            row = tx.execute("SELECT throttle_json,decision_json FROM resource_leases WHERE id=?", (lease_id,)).fetchone()
            if not row:
                raise KeyError(lease_id)
            tx.execute(
                """UPDATE resource_leases SET status=?,throttle_json=?,decision_json=?,released_at=? WHERE id=?""",
                (
                    status,
                    canonical_json(throttle) if throttle is not None else row["throttle_json"],
                    canonical_json(decision) if decision is not None else row["decision_json"],
                    released_at,
                    lease_id,
                ),
            )

    def get_resource_lease(self, lease_id: str) -> dict[str, Any] | None:
        row = self._connection().execute("SELECT * FROM resource_leases WHERE id=?", (lease_id,)).fetchone()
        return self._lease_from_row(row) if row else None

    def resource_leases(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM resource_leases"
        if active_only:
            query += " WHERE status IN ('acquiring','active','releasing','unknown')"
        query += " ORDER BY created_at,id"
        return [self._lease_from_row(row) for row in self._connection().execute(query).fetchall()]

    def mark_active_leases_unknown(self) -> list[dict[str, Any]]:
        with self.transaction(immediate=True) as tx:
            tx.execute(
                "UPDATE resource_leases SET status='unknown' WHERE status IN ('acquiring','active','releasing')"
            )
            rows = tx.execute("SELECT * FROM resource_leases WHERE status='unknown' ORDER BY created_at,id").fetchall()
        return [self._lease_from_row(row) for row in rows]

    def submit(self, request: dict[str, Any], *, max_pending_jobs: int | None = None) -> tuple[dict[str, Any], bool]:
        digest = request_digest(request)
        now = utc_now()
        job_id = f"job_{uuid.uuid4().hex}"
        with self.transaction(immediate=True) as tx:
            existing = tx.execute(
                "SELECT * FROM jobs WHERE origin=? AND idempotency_key=?",
                (request["origin"], request["idempotencyKey"]),
            ).fetchone()
            if existing:
                if existing["request_digest"] != digest:
                    raise IdempotencyConflict("idempotency key was already used with a different request")
                return self._job_from_row(existing), False
            if max_pending_jobs is not None:
                pending = tx.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued','loading','running','validating')"
                ).fetchone()["count"]
                if int(pending) >= max_pending_jobs:
                    raise QueueFull("broker pending-job limit reached")
            tx.execute(
                """INSERT INTO jobs(
                    id,origin,idempotency_key,request_id,trace_id,capability,priority,status,
                    request_digest,request_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id, request["origin"], request["idempotencyKey"], request["requestId"],
                    request["traceId"], request["capability"], request["priority"], "queued",
                    digest, canonical_json(request), now, now,
                ),
            )
            tx.execute(
                "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                (job_id, "queued", "{}", now),
            )
            row = tx.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_from_row(row), True

    def artifact_usage_bytes(self) -> int:
        row = self._connection().execute("SELECT COALESCE(SUM(size_bytes),0) AS total FROM artifacts").fetchone()
        return int(row["total"])

    def claim_next(self, supported: set[str]) -> dict[str, Any] | None:
        if not supported:
            return None
        now = utc_now()
        placeholders = ",".join("?" for _ in supported)
        with self.transaction(immediate=True) as tx:
            row = tx.execute(
                f"""SELECT * FROM jobs
                    WHERE status='queued' AND cancel_requested=0
                      AND (not_before IS NULL OR not_before<=?)
                      AND capability IN ({placeholders})
                    ORDER BY priority DESC, created_at ASC LIMIT 1""",
                (now, *tuple(sorted(supported))),
            ).fetchone()
            if not row:
                return None
            updated = tx.execute(
                "UPDATE jobs SET status='loading', not_before=NULL, attempt=attempt+1, started_at=COALESCE(started_at,?), updated_at=? WHERE id=? AND status='queued' AND cancel_requested=0",
                (now, now, row["id"]),
            )
            if updated.rowcount != 1:
                return None
            tx.execute(
                "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                (row["id"], "loading", "{}", now),
            )
            claimed = tx.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        return self._job_from_row(claimed)

    def peek_next(self, supported: set[str]) -> dict[str, Any] | None:
        values = self.peek_queued(supported, limit=1)
        return values[0] if values else None

    def peek_queued(self, supported: set[str], *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        if not supported:
            return []
        if not 1 <= limit <= 100:
            raise ValueError("peek limit must be 1-100")
        if offset < 0:
            raise ValueError("peek offset must be non-negative")
        now = utc_now()
        placeholders = ",".join("?" for _ in supported)
        rows = self._connection().execute(
            f"""SELECT * FROM jobs
                WHERE status='queued' AND cancel_requested=0
                  AND (not_before IS NULL OR not_before<=?)
                  AND capability IN ({placeholders})
                ORDER BY priority DESC, created_at ASC LIMIT ? OFFSET ?""",
            (now, *tuple(sorted(supported)), limit, offset),
        ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def defer(self, job_id: str, *, detail: dict[str, Any], delay_seconds: float = 2.0) -> None:
        now = utc_now()
        not_before = (datetime.now(timezone.utc) + timedelta(seconds=max(0.1, min(delay_seconds, 300)))).isoformat().replace("+00:00", "Z")
        with self.transaction(immediate=True) as tx:
            updated = tx.execute(
                "UPDATE jobs SET status='queued',not_before=?,updated_at=? WHERE id=? AND status IN ('loading','running') AND cancel_requested=0",
                (not_before, now, job_id),
            )
            if updated.rowcount:
                tx.execute(
                    "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                    (job_id, "queued", canonical_json({"phase": "admission_wait", **detail}), now),
                )

    def transition(self, job_id: str, status: str, *, detail: dict[str, Any] | None = None, profile_id: str | None = None) -> None:
        now = utc_now()
        terminal = status in {"completed", "failed", "cancelled", "interrupted"}
        with self.transaction(immediate=True) as tx:
            tx.execute(
                "UPDATE jobs SET status=?, profile_id=COALESCE(?,profile_id), updated_at=?, finished_at=CASE WHEN ? THEN ? ELSE finished_at END WHERE id=?",
                (status, profile_id, now, int(terminal), now, job_id),
            )
            tx.execute(
                "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                (job_id, status, canonical_json(detail or {}), now),
            )

    def finish(self, job_id: str, result: dict[str, Any]) -> bool:
        now = utc_now()
        with self.transaction(immediate=True) as tx:
            updated = tx.execute(
                """UPDATE jobs SET status='completed', result_json=?, error_json=NULL, updated_at=?, finished_at=?
                   WHERE id=? AND cancel_requested=0 AND status IN ('loading','running','validating')""",
                (canonical_json(result), now, now, job_id),
            )
            if updated.rowcount != 1:
                row = tx.execute("SELECT cancel_requested,status FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row and row["cancel_requested"] and row["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    cancellation = {"code": "cancelled", "message": "job was cancelled before completion committed", "retryable": False}
                    tx.execute(
                        "UPDATE jobs SET status='cancelled',error_json=?,updated_at=?,finished_at=? WHERE id=?",
                        (canonical_json(cancellation), now, now, job_id),
                    )
                    tx.execute(
                        "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                        (job_id, "cancelled", canonical_json(cancellation), now),
                    )
                return False
            tx.execute(
                "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                (job_id, "completed", canonical_json({"artifactCount": len(result.get("artifacts", []))}), now),
            )
        return True

    def fail(self, job_id: str, error: dict[str, Any], *, status: str = "failed") -> None:
        now = utc_now()
        with self.transaction(immediate=True) as tx:
            tx.execute(
                "UPDATE jobs SET status=?, error_json=?, updated_at=?, finished_at=? WHERE id=?",
                (status, canonical_json(error), now, now, job_id),
            )
            tx.execute(
                "INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)",
                (job_id, status, canonical_json(error), now),
            )

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with self.transaction(immediate=True) as tx:
            row = tx.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] == "queued":
                tx.execute("UPDATE jobs SET status='cancelled',cancel_requested=1,updated_at=?,finished_at=? WHERE id=?", (now, now, job_id))
                tx.execute("INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)", (job_id, "cancelled", canonical_json({"reason": "cancelled_before_start"}), now))
            elif row["status"] not in {"completed", "failed", "cancelled", "interrupted"}:
                tx.execute("UPDATE jobs SET cancel_requested=1,updated_at=? WHERE id=?", (now, job_id))
                tx.execute("INSERT INTO job_events(job_id,status,detail_json,created_at) VALUES(?,?,?,?)", (job_id, row["status"], canonical_json({"cancelRequested": True}), now))
            updated = tx.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_from_row(updated)

    def cancel_requested(self, job_id: str) -> bool:
        row = self._connection().execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self._connection().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._job_from_row(row) if row else None

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = self._connection().execute(
            "SELECT * FROM job_events WHERE job_id=? AND sequence>? ORDER BY sequence", (job_id, after)
        ).fetchall()
        return [
            {"sequence": row["sequence"], "status": row["status"], "detail": json.loads(row["detail_json"]), "createdAt": row["created_at"]}
            for row in rows
        ]

    def add_artifact(self, artifact: dict[str, Any], relative_path: str) -> None:
        self._connection().execute(
            """INSERT INTO artifacts(id,job_id,kind,role,media_type,relative_path,sha256,size_bytes,metadata_json,validation_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                artifact["id"], artifact.get("jobId"), artifact["kind"], artifact["role"], artifact["mediaType"],
                relative_path, artifact["sha256"], artifact["sizeBytes"], canonical_json(artifact.get("metadata", {})),
                canonical_json(artifact.get("validation", {})), artifact["createdAt"],
            ),
        )

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self._connection().execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return self._artifact_from_row(row) if row else None

    def job_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        rows = self._connection().execute("SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at,id", (job_id,)).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": row["id"], "requestId": row["request_id"], "traceId": row["trace_id"],
            "origin": row["origin"], "idempotencyKey": row["idempotency_key"], "capability": row["capability"],
            "priority": row["priority"], "status": row["status"], "profileId": row["profile_id"],
            "cancelRequested": bool(row["cancel_requested"]), "attempt": row["attempt"],
            "createdAt": row["created_at"], "updatedAt": row["updated_at"],
            "startedAt": row["started_at"], "finishedAt": row["finished_at"],
            "request": json.loads(row["request_json"]),
        }
        if row["result_json"]:
            value["result"] = json.loads(row["result_json"])
        if row["error_json"]:
            value["error"] = json.loads(row["error_json"])
        return value

    @staticmethod
    def _artifact_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "jobId": row["job_id"], "kind": row["kind"], "role": row["role"],
            "mediaType": row["media_type"], "sha256": row["sha256"], "sizeBytes": row["size_bytes"],
            "metadata": json.loads(row["metadata_json"]), "validation": json.loads(row["validation_json"]),
            "createdAt": row["created_at"], "_relativePath": row["relative_path"],
        }

    @staticmethod
    def _lease_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "jobId": row["job_id"],
            "brokerEpoch": row["broker_epoch"],
            "fencingToken": row["fencing_token"],
            "resourceGroup": row["resource_group"],
            "profileId": row["profile_id"],
            "routeId": row["route_id"],
            "serviceClass": row["service_class"],
            "mode": row["mode"],
            "estimatedMemoryGb": row["estimated_memory_gb"],
            "status": row["status"],
            "throttle": json.loads(row["throttle_json"]),
            "decision": json.loads(row["decision_json"]),
            "createdAt": row["created_at"],
            "releasedAt": row["released_at"],
        }
