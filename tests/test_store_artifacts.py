from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from spark_broker.artifacts import ArtifactError, ArtifactRegistry
from spark_broker.contract import validate_job_request
from spark_broker.store import IdempotencyConflict, QueueFull, Store
from tests.helpers import request


class StoreArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "broker.sqlite3")
        self.registry = ArtifactRegistry(self.root / "artifacts", self.store, max_upload_bytes=1024 * 1024)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_close_releases_the_calling_threads_connection(self) -> None:
        connection = self.store._connection()
        self.store.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_idempotency_replays_same_request_and_rejects_different_body(self) -> None:
        value = validate_job_request(request(idempotencyKey="stable-key"), broker_id="spark.test")
        first, created = self.store.submit(value)
        second, replay_created = self.store.submit(value)
        self.assertTrue(created)
        self.assertFalse(replay_created)
        self.assertEqual(first["id"], second["id"])
        changed = {**value, "priority": 99}
        with self.assertRaises(IdempotencyConflict):
            self.store.submit(changed)

    def test_restart_marks_active_jobs_interrupted_but_leaves_queued(self) -> None:
        active_request = validate_job_request(request(idempotencyKey="active"), broker_id="spark.test")
        queued_request = validate_job_request(request(idempotencyKey="queued"), broker_id="spark.test")
        active, _ = self.store.submit(active_request)
        queued, _ = self.store.submit(queued_request)
        claimed = self.store.claim_next({"system.echo"})
        self.assertEqual(claimed["id"], active["id"])
        self.assertEqual(claimed["status"], "loading")
        restarted = Store(self.root / "broker.sqlite3")
        try:
            self.assertEqual(restarted.get_job(active["id"])["status"], "loading")
            restarted.interrupt_uncertain_jobs()
            self.assertEqual(restarted.get_job(active["id"])["status"], "interrupted")
            self.assertFalse(restarted.get_job(active["id"])["error"]["retryable"])
            self.assertEqual(restarted.get_job(queued["id"])["status"], "queued")
        finally:
            restarted.close()

    def test_artifact_commit_is_hashed_and_integrity_checked(self) -> None:
        payload = b"valid artifact content"
        artifact = self.registry.import_stream(io.BytesIO(payload), size=len(payload), kind="image", role="source", media_type="image/png")
        public, path = self.registry.resolve(artifact["id"], verify=True)
        self.assertEqual(public["sha256"], artifact["sha256"])
        self.assertEqual(path.read_bytes(), payload)
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ArtifactError, "integrity"):
            self.registry.resolve(artifact["id"], verify=True)

    def test_upload_hash_mismatch_leaves_no_registered_artifact(self) -> None:
        with self.assertRaisesRegex(ArtifactError, "hash"):
            self.registry.import_stream(io.BytesIO(b"abc"), size=3, kind="image", role="source", media_type="image/png", expected_sha256="0" * 64)
        count = self.store._connection().execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        self.assertEqual(count, 0)
        self.assertEqual(list((self.root / "artifacts" / ".staging").iterdir()), [])

    def test_aggregate_artifact_quota_fails_before_registration(self) -> None:
        registry = ArtifactRegistry(
            self.root / "quota-artifacts", self.store,
            max_upload_bytes=1024, max_storage_bytes=5,
        )
        registry.import_stream(io.BytesIO(b"abc"), size=3, kind="text", role="first", media_type="text/plain")
        with self.assertRaisesRegex(ArtifactError, "quota"):
            registry.import_stream(io.BytesIO(b"def"), size=3, kind="text", role="second", media_type="text/plain")
        self.assertEqual(self.store.artifact_usage_bytes(), 3)

    def test_pending_job_limit_is_transactional_and_replays_still_work(self) -> None:
        first_request = validate_job_request(request(idempotencyKey="queue-first"), broker_id="spark.test")
        first, _ = self.store.submit(first_request, max_pending_jobs=1)
        replay, created = self.store.submit(first_request, max_pending_jobs=1)
        self.assertFalse(created)
        self.assertEqual(replay["id"], first["id"])
        second_request = validate_job_request(request(idempotencyKey="queue-second"), broker_id="spark.test")
        with self.assertRaises(QueueFull):
            self.store.submit(second_request, max_pending_jobs=1)

    def test_deferred_high_priority_job_does_not_starve_runnable_work(self) -> None:
        high_request = validate_job_request(
            request(idempotencyKey="deferred-high", priority=100), broker_id="spark.test"
        )
        high, _ = self.store.submit(high_request)
        self.assertEqual(self.store.claim_next({"system.echo"})["id"], high["id"])
        self.store.defer(high["id"], detail={"code": "resource_busy"}, delay_seconds=60)
        low_request = validate_job_request(
            request(idempotencyKey="runnable-low", priority=10), broker_id="spark.test"
        )
        low, _ = self.store.submit(low_request)
        self.assertEqual(self.store.claim_next({"system.echo"})["id"], low["id"])

    def test_cancel_queued_job_is_terminal_and_idempotent(self) -> None:
        value = validate_job_request(request(), broker_id="spark.test")
        job, _ = self.store.submit(value)
        self.assertEqual(self.store.cancel(job["id"])["status"], "cancelled")

    def test_cancel_requested_wins_race_against_completion_commit(self) -> None:
        value = validate_job_request(request(idempotencyKey="cancel-race"), broker_id="spark.test")
        job, _ = self.store.submit(value)
        claimed = self.store.claim_next({"system.echo"})
        self.assertEqual(claimed["id"], job["id"])
        self.store.transition(job["id"], "running")
        self.assertTrue(self.store.cancel(job["id"])["cancelRequested"])
        committed = self.store.finish(job["id"], {"artifacts": [], "continuations": [], "data": {}})
        self.assertFalse(committed)
        current = self.store.get_job(job["id"])
        self.assertEqual(current["status"], "cancelled")
        self.assertNotIn("result", current)
        self.assertEqual(self.store.cancel(job["id"])["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
