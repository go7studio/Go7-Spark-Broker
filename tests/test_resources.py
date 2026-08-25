from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spark_broker.contract import validate_job_request
from spark_broker.resources import (
    AdmissionDeferred,
    ControllerPolicy,
    ExecutionPlan,
    ResourceCoordinator,
    ResourcePolicy,
)
from spark_broker.store import Store
from tests.helpers import request


class FakeHostProbe:
    def __init__(self, available_gb: int = 120, pressure: float = 0.0) -> None:
        self.available_gb = available_gb
        self.pressure = pressure

    def snapshot(self):
        return {
            "availableMemoryBytes": self.available_gb * 1024**3,
            "totalMemoryBytes": 128 * 1024**3,
            "swapFreeBytes": 0,
            "memoryPressureAvg10": self.pressure,
            "sampledAtMonotonic": 1.0,
        }


class FakeControlClient:
    def __init__(self, *, unknown_consumers: int = 0) -> None:
        self.unknown_consumers = unknown_consumers
        self.calls: list[dict] = []

    def snapshot(self, endpoint, token_file):
        del endpoint, token_file
        return {"health": "healthy", "unknownConsumers": self.unknown_consumers, "activeProfiles": [], "generation": 7}

    def set_mode(self, controller, *, mode, lease_id, fencing_token, generation, reason):
        call = {
            "controller": controller.id, "mode": mode, "leaseId": lease_id,
            "fencingToken": fencing_token, "generation": generation, "reason": reason,
        }
        self.calls.append(call)
        return {
            "leaseId": lease_id,
            "fencingToken": fencing_token,
            "acknowledgedGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
        }


class ResourceCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "broker.sqlite3")
        submitted = validate_job_request(
            request(capability="text.chat.generate", idempotencyKey="resource-test"),
            broker_id="spark.test",
        )
        self.job, _ = self.store.submit(submitted)
        self.token = self.root / "controller-token"
        self.token.write_text("t" * 32, encoding="utf-8")
        self.token.chmod(0o600)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def policy(self, *, require_probe: bool = True) -> ResourcePolicy:
        return ResourcePolicy(
            host_reserve_gb=16,
            maximum_memory_pressure_avg10=5,
            enforce_memory_admission=True,
            require_probe=require_probe,
            probe_endpoint="http://127.0.0.1:9000",
            probe_token_file=self.token,
            controllers=(ControllerPolicy(
                id="background-service",
                profile_id="gpu.background",
                endpoint="http://127.0.0.1:9001",
                token_file=self.token,
                throttle_for=("interactive",),
                normal_mode="normal",
                throttled_mode="interactive-boost",
                priority=20,
            ),),
        )

    @staticmethod
    def plan() -> ExecutionPlan:
        return ExecutionPlan(
            profile_id="gpu.interactive",
            route_id="interactive-small",
            resource_group="gpu:0",
            service_class="interactive",
            lease_mode="exclusive",
            estimated_memory_gb=32,
            route_reason="policy:latency",
        )

    def test_throttle_is_fenced_journaled_and_restored_before_release(self) -> None:
        control = FakeControlClient()
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        coordinator.start()
        try:
            handle = coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertIsNotNone(handle)
            active = self.store.resource_leases(active_only=True)
            self.assertEqual(active[0]["status"], "active")
            self.assertEqual(active[0]["routeId"], "interactive-small")
            self.assertTrue(active[0]["fencingToken"].startswith(f"fence_{coordinator.epoch}_"))
            coordinator.release(handle)
        finally:
            coordinator.stop()
        self.assertEqual([item["mode"] for item in control.calls], ["interactive-boost", "normal"])
        self.assertEqual(control.calls[0]["leaseId"], control.calls[1]["leaseId"])
        self.assertEqual(control.calls[0]["fencingToken"], control.calls[1]["fencingToken"])
        self.assertEqual(self.store.resource_leases()[0]["status"], "released")

    def test_unknown_gpu_consumer_fails_closed_without_mutating_it(self) -> None:
        control = FakeControlClient(unknown_consumers=1)
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "unknown_gpu_consumer")
            self.assertEqual(self.store.resource_leases(), [])
            self.assertEqual(control.calls, [])
        finally:
            coordinator.stop()

    def test_memory_envelope_includes_host_reserve(self) -> None:
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(available_gb=40), control_client=FakeControlClient(),
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "insufficient_memory")
        finally:
            coordinator.stop()

    def test_os_lock_prevents_two_mutating_coordinators(self) -> None:
        first = ResourceCoordinator(store=self.store, data_root=self.root)
        second_store = Store(self.root / "broker.sqlite3")
        second = ResourceCoordinator(store=second_store, data_root=self.root)
        first.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "another broker instance"):
                second.start()
        finally:
            first.stop()
            second_store.close()


if __name__ == "__main__":
    unittest.main()
