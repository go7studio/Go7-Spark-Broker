from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from spark_broker.artifacts import ArtifactRegistry
from spark_broker.executors import Capability, EchoExecutor, Executor
from spark_broker.scheduler import Scheduler
from spark_broker.store import Store
from spark_broker.contract import validate_job_request
from spark_broker.resources import AdmissionDeferred, ExecutionControl, ExecutionPlan, LeaseHandle
from tests.helpers import request


class RecordingExecutor(Executor):
    def __init__(self, capability_id: str, profile_id: str, record: list[str]) -> None:
        self.capability = Capability(capability_id, profile_id, "test", (), (), 1)
        self.record = record

    def activate(self, cancelled):
        self.record.append(f"activate:{self.capability.profile_id}")

    def execute(self, job, registry, cancelled, stage):
        stage("running", {"profileId": self.capability.profile_id})
        self.record.append(f"run:{self.capability.profile_id}")
        return {"artifacts": [], "continuations": [], "data": {"profileId": self.capability.profile_id}}


class RejectingExecutor(RecordingExecutor):
    def validate_request(self, job, registry):
        del job, registry
        raise ValueError("request rejected before activation")


class YieldingTrainingExecutor(Executor):
    def __init__(self, started: threading.Event) -> None:
        self.capability = Capability(
            "model.training.run", "gpu.training", "test training", (), (), 32,
            resource_group="gpu:0", service_class="training", lease_mode="exclusive", preemption_mode="checkpoint",
        )
        self.started = started
        self.saw_yield = False

    def execute(self, job, registry, cancelled, stage):
        del job, registry
        stage("running", {"phase": "training"})
        self.started.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if getattr(cancelled, "yield_requested", lambda: False)():
                self.saw_yield = True
                return {"artifacts": [], "continuations": [], "data": {"outcome": "yielded"}}
            time.sleep(0.01)
        return {"artifacts": [], "continuations": [], "data": {"outcome": "completed"}}


class InteractiveEcho(EchoExecutor):
    capability = Capability(
        "text.interactive.test", "cpu.interactive", "interactive test", (), (), 0,
        resource_group=None, service_class="interactive", lease_mode="none",
    )


class PartialActivationExecutor(Executor):
    capability = Capability(
        "runtime.partial.test", "cpu.partial", "partial activation test", (), (), 0,
        resource_group=None, service_class="batch", lease_mode="none",
    )

    def __init__(self) -> None:
        self.deactivated = False

    def activate(self, cancelled):
        del cancelled
        raise RuntimeError("readiness failed after partial activation")

    def deactivate_plan(self, plan):
        del plan
        self.deactivated = True
        return True

    def execute(self, job, registry, cancelled, stage):
        raise AssertionError("execute must not run")


class DeferringExecutor(Executor):
    capability = Capability(
        "gpu.deferred.test", "gpu.deferred", "deferred test", (), (), 1,
        resource_group="gpu:0", service_class="interactive", lease_mode="exclusive",
    )

    def activate(self, cancelled):
        del cancelled
        raise AdmissionDeferred("resource_busy", "resource is temporarily busy", retry_after_seconds=30)

    def execute(self, job, registry, cancelled, stage):
        raise AssertionError("execute must not run after deferred activation")


class ReleaseFailingCoordinator:
    def __init__(self) -> None:
        self.release_calls = 0

    def routing_context(self):
        return {}

    def acquire(self, job, plan, cancelled):
        del cancelled
        return LeaseHandle(lease={
            "id": "lease_test", "jobId": job["id"], "mode": plan.lease_mode,
            "profileId": plan.profile_id, "fencingToken": "fence_test",
        })

    def verify_activation(self, handle, plan):
        del handle, plan

    def configure_execution_control(self, plan, control):
        del plan, control

    def release(self, handle):
        del handle
        self.release_calls += 1
        raise AdmissionDeferred("release_unverified", "release could not be verified")


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = Store(root / "db.sqlite3")
        self.registry = ArtifactRegistry(root / "artifacts", self.store, max_upload_bytes=1024)
        executor = EchoExecutor()
        self.scheduler = Scheduler(broker_id="spark.test", store=self.store, registry=self.registry, executors={executor.capability.id: executor})
        self.scheduler.start()

    def tearDown(self) -> None:
        self.scheduler.stop()
        self.store.close()
        self.temporary.cleanup()

    def wait(self, job_id: str) -> dict:
        for _ in range(100):
            job = self.store.get_job(job_id)
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return job
            time.sleep(0.02)
        self.fail("job did not finish")

    def test_echo_runs_through_durable_state_machine(self) -> None:
        value = validate_job_request(request(metadata={"message": "hello"}), broker_id="spark.test")
        job, _ = self.store.submit(value)
        self.scheduler.notify()
        finished = self.wait(job["id"])
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(finished["result"]["data"]["echo"], {"message": "hello"})
        self.assertEqual(finished["result"]["route"], {"visitedSystems": ["spark.test"], "hopCount": 1})
        states = [event["status"] for event in self.store.events(job["id"])]
        self.assertEqual(states, ["queued", "loading", "loading", "running", "completed"])

    def test_scheduler_rotates_text_to_3d_and_back(self) -> None:
        self.scheduler.stop()
        record: list[str] = []
        text = RecordingExecutor("text.chat.generate", "gpu.text-runtime", record)
        model = RecordingExecutor("asset.3d.generate", "gpu.3d-runtime", record)
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={text.capability.id: text, model.capability.id: model},
        )
        jobs = []
        for capability, priority, key in (
            ("text.chat.generate", 90, "rotation-text-1"),
            ("asset.3d.generate", 80, "rotation-3d"),
            ("text.chat.generate", 70, "rotation-text-2"),
        ):
            value = validate_job_request(request(capability=capability, priority=priority, idempotencyKey=key), broker_id="spark.test")
            jobs.append(self.store.submit(value)[0])
        scheduler.start()
        scheduler.notify()
        try:
            for job in jobs:
                for _ in range(100):
                    if self.store.get_job(job["id"])["status"] == "completed":
                        break
                    time.sleep(0.02)
                self.assertEqual(self.store.get_job(job["id"])["status"], "completed")
        finally:
            scheduler.stop()
        self.assertEqual(record, [
            "activate:gpu.text-runtime", "run:gpu.text-runtime",
            "activate:gpu.3d-runtime", "run:gpu.3d-runtime",
            "activate:gpu.text-runtime", "run:gpu.text-runtime",
        ])

    def test_scheduler_revalidates_a_same_profile_before_each_job(self) -> None:
        self.scheduler.stop()
        record: list[str] = []
        text = RecordingExecutor("text.chat.generate", "gpu.text-runtime", record)
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={text.capability.id: text},
        )
        jobs = []
        for key in ("same-profile-1", "same-profile-2"):
            value = validate_job_request(request(capability="text.chat.generate", idempotencyKey=key), broker_id="spark.test")
            jobs.append(self.store.submit(value)[0])
        scheduler.start()
        scheduler.notify()
        try:
            for job in jobs:
                for _ in range(100):
                    if self.store.get_job(job["id"])["status"] == "completed":
                        break
                    time.sleep(0.02)
                self.assertEqual(self.store.get_job(job["id"])["status"], "completed")
        finally:
            scheduler.stop()
        self.assertEqual(record, [
            "activate:gpu.text-runtime", "run:gpu.text-runtime",
            "activate:gpu.text-runtime", "run:gpu.text-runtime",
        ])

    def test_validation_failure_happens_before_activation(self) -> None:
        self.scheduler.stop()
        record: list[str] = []
        rejecting = RejectingExecutor("asset.invalid.generate", "gpu.invalid", record)
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={rejecting.capability.id: rejecting},
        )
        value = validate_job_request(request(capability=rejecting.capability.id), broker_id="spark.test")
        job, _ = self.store.submit(value)
        scheduler.start()
        scheduler.notify()
        try:
            finished = self.wait(job["id"])
        finally:
            scheduler.stop()
        self.assertEqual(finished["status"], "failed")
        self.assertEqual(record, [])

    def test_control_loop_observes_interactive_queue_during_training(self) -> None:
        self.scheduler.stop()
        started = threading.Event()
        training = YieldingTrainingExecutor(started)
        echo = InteractiveEcho()
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={training.capability.id: training, echo.capability.id: echo},
        )
        training_request = validate_job_request(request(
            capability=training.capability.id, priority=10, idempotencyKey="training-control-test",
        ), broker_id="spark.test")
        training_job, _ = self.store.submit(training_request)
        scheduler.start()
        scheduler.notify()
        self.assertTrue(started.wait(1))
        interactive_request = validate_job_request(request(
            capability=echo.capability.id, priority=90, idempotencyKey="interactive-control-test",
        ), broker_id="spark.test")
        interactive_job, _ = self.store.submit(interactive_request)
        scheduler.notify()
        try:
            training_result = self.wait(training_job["id"])
            interactive_result = self.wait(interactive_job["id"])
        finally:
            scheduler.stop()
        self.assertTrue(training.saw_yield)
        self.assertEqual(training_result["status"], "failed")
        self.assertEqual(training_result["error"]["code"], "yield_protocol_unsupported")
        self.assertEqual(interactive_result["status"], "completed")

    def test_stop_refuses_to_release_ownership_while_worker_is_alive(self) -> None:
        self.scheduler.stop()
        release = threading.Event()
        blocked = threading.Thread(target=release.wait, name="blocked-scheduler", daemon=True)
        blocked.start()
        self.scheduler._thread = blocked
        self.scheduler._control_thread = None
        try:
            with self.assertRaisesRegex(RuntimeError, "ownership is retained"):
                self.scheduler.stop(timeout=0.01)
        finally:
            release.set()
            blocked.join(1)

    def test_release_failure_never_requeues_an_already_executed_job(self) -> None:
        self.scheduler.stop()
        coordinator = ReleaseFailingCoordinator()
        executor = EchoExecutor()
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={executor.capability.id: executor}, coordinator=coordinator,  # type: ignore[arg-type]
        )
        value = validate_job_request(
            request(idempotencyKey="release-uncertain-test"), broker_id="spark.test"
        )
        job, _ = self.store.submit(value)
        scheduler.start()
        scheduler.notify()
        try:
            finished = self.wait(job["id"])
        finally:
            scheduler.stop()
        self.assertEqual(finished["status"], "failed")
        self.assertEqual(finished["error"]["code"], "resource_release_uncertain")
        self.assertFalse(finished["error"]["retryable"])
        self.assertGreaterEqual(coordinator.release_calls, 1)

    def test_partial_activation_is_compensated(self) -> None:
        self.scheduler.stop()
        executor = PartialActivationExecutor()
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={executor.capability.id: executor},
        )
        value = validate_job_request(
            request(capability=executor.capability.id, idempotencyKey="partial-activation"),
            broker_id="spark.test",
        )
        job, _ = self.store.submit(value)
        scheduler.start()
        scheduler.notify()
        try:
            finished = self.wait(job["id"])
        finally:
            scheduler.stop()
        self.assertEqual(finished["status"], "failed")
        self.assertTrue(executor.deactivated)

    def test_deferred_job_does_not_block_other_runnable_work(self) -> None:
        self.scheduler.stop()
        deferred = DeferringExecutor()
        echo = EchoExecutor()
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={deferred.capability.id: deferred, echo.capability.id: echo},
        )
        blocked_value = validate_job_request(request(
            capability=deferred.capability.id, priority=100,
            idempotencyKey="deferred-does-not-block",
        ), broker_id="spark.test")
        echo_value = validate_job_request(request(
            priority=10, idempotencyKey="runnable-behind-deferred",
        ), broker_id="spark.test")
        blocked_job, _ = self.store.submit(blocked_value)
        echo_job, _ = self.store.submit(echo_value)
        scheduler.start()
        scheduler.notify()
        try:
            finished = self.wait(echo_job["id"])
        finally:
            scheduler.stop()
        self.assertEqual(finished["status"], "completed")
        self.assertEqual(self.store.get_job(blocked_job["id"])["status"], "queued")

    def test_control_loop_failure_stops_main_claiming_loop(self) -> None:
        self.scheduler.stop()
        executor = EchoExecutor()
        scheduler = Scheduler(
            broker_id="spark.test", store=self.store, registry=self.registry,
            executors={executor.capability.id: executor},
        )
        scheduler.start()
        with scheduler._state_lock:
            scheduler._active_priority = 1
            scheduler._active_plan = ExecutionPlan(
                profile_id="gpu.training", route_id=None, resource_group="gpu:0",
                service_class="training", lease_mode="exclusive", estimated_memory_gb=1,
                preemption_mode="checkpoint",
            )
            scheduler._active_control = ExecutionControl(lambda: False)
        original = self.store.peek_queued

        def broken(*args, **kwargs):
            raise RuntimeError("database failure")

        self.store.peek_queued = broken  # type: ignore[method-assign]
        scheduler.notify()
        try:
            for _ in range(100):
                if not scheduler.status()["controlLoopAlive"]:
                    break
                time.sleep(0.01)
            self.assertFalse(scheduler.status()["controlLoopAlive"])
            self.assertTrue(scheduler._stop.is_set())
        finally:
            self.store.peek_queued = original  # type: ignore[method-assign]
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()
