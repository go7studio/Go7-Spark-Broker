from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from spark_broker.contract import validate_job_request
from spark_broker.resources import (
    AdmissionDeferred,
    ControllerPolicy,
    ExecutionControl,
    ExecutionPlan,
    ResourceControlClient,
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
    def __init__(self, *, unknown_consumers: int = 0, active_profiles: list[str] | None = None) -> None:
        self.unknown_consumers = unknown_consumers
        self.active_profiles = active_profiles or []
        self.calls: list[dict] = []

    def snapshot(self, endpoint, token_file):
        del endpoint, token_file
        return {
            "health": "healthy", "unknownConsumers": self.unknown_consumers,
            "activeProfiles": self.active_profiles, "profiles": {}, "generation": 7,
        }

    def set_mode(self, controller, *, mode, lease_id, fencing_token, broker_epoch, generation, reason):
        call = {
            "controller": controller.id, "mode": mode, "leaseId": lease_id,
            "fencingToken": fencing_token, "brokerEpoch": broker_epoch,
            "generation": generation, "reason": reason,
        }
        self.calls.append(call)
        return {
            "leaseId": lease_id,
            "fencingToken": fencing_token,
            "brokerEpoch": broker_epoch,
            "acknowledgedGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
            "snapshotGeneration": 7,
        }

    def takeover_mode(
        self, controller, *, previous_lease_id, previous_fencing_token,
        recovery_fencing_token, broker_epoch, generation, mode,
    ):
        self.calls.append({
            "controller": controller.id, "mode": mode,
            "previousLeaseId": previous_lease_id,
            "previousFencingToken": previous_fencing_token,
            "recoveryFencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "generation": generation,
        })
        return {
            "previousLeaseId": previous_lease_id,
            "previousFencingToken": previous_fencing_token,
            "recoveryFencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "acknowledgedGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
            "snapshotGeneration": 7,
        }


class FailingRestoreControlClient(FakeControlClient):
    def set_mode(self, controller, *, mode, lease_id, fencing_token, broker_epoch, generation, reason):
        if mode == controller.normal_mode:
            raise AdmissionDeferred("restore_rejected", "controller rejected restore")
        return super().set_mode(
            controller, mode=mode, lease_id=lease_id, fencing_token=fencing_token,
            broker_epoch=broker_epoch, generation=generation, reason=reason,
        )


class JournalCheckingControlClient(FakeControlClient):
    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store

    def set_mode(self, controller, *, mode, lease_id, fencing_token, broker_epoch, generation, reason):
        if mode == controller.throttled_mode:
            lease = self.store.get_resource_lease(lease_id)
            assert lease is not None
            assert lease["throttle"][-1]["state"] == "requested"
            assert lease["throttle"][-1]["controllerId"] == controller.id
        return super().set_mode(
            controller, mode=mode, lease_id=lease_id, fencing_token=fencing_token,
            broker_epoch=broker_epoch, generation=generation, reason=reason,
        )


class StaleSnapshotControlClient(FakeControlClient):
    def set_mode(self, *args, **kwargs):
        value = super().set_mode(*args, **kwargs)
        value["snapshotGeneration"] = 8
        return value


class StubResourceControlClient(ResourceControlClient):
    def __init__(self, response: dict) -> None:
        super().__init__()
        self.response = response

    def _request(self, method, url, token_file, payload):
        del method, url, token_file, payload
        return dict(self.response)


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
        control = JournalCheckingControlClient(self.store)
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
        self.assertEqual(self.store.resource_leases()[0]["throttle"][0]["state"], "applied")

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

    def test_explicit_host_lock_prevents_brokers_with_different_data_roots(self) -> None:
        other_root = self.root / "other"
        other_store = Store(other_root / "broker.sqlite3")
        lock_path = self.root / "host-gpu0.lock"
        first = ResourceCoordinator(store=self.store, data_root=self.root, lock_path=lock_path)
        second = ResourceCoordinator(store=other_store, data_root=other_root, lock_path=lock_path)
        first.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "another broker instance"):
                second.start()
        finally:
            first.stop()
        second.start()
        try:
            self.assertGreater(second.epoch, first.epoch)
        finally:
            second.stop()
            other_store.close()

    def test_host_lock_rejects_non_regular_file(self) -> None:
        lock_path = self.root / "host-gpu0.lock"
        os.mkfifo(lock_path, 0o600)
        coordinator = ResourceCoordinator(store=self.store, data_root=self.root, lock_path=lock_path)
        with self.assertRaisesRegex(RuntimeError, "regular file"):
            coordinator.start()

    def test_host_epoch_rejects_symlink(self) -> None:
        epoch_target = self.root / "epoch-target"
        epoch_target.write_text("1\n", encoding="ascii")
        epoch_target.chmod(0o600)
        epoch_path = self.root / "host-gpu0.epoch"
        epoch_path.symlink_to(epoch_target)
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            lock_path=self.root / "host-gpu0.lock",
            epoch_path=epoch_path,
        )
        with self.assertRaisesRegex(RuntimeError, "epoch file cannot be opened safely"):
            coordinator.start()

    def test_failed_admission_rollback_is_unknown_and_quarantines(self) -> None:
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(available_gb=40), control_client=FailingRestoreControlClient(),
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "admission_rollback_failed")
            self.assertTrue(coordinator.quarantined)
            self.assertEqual(self.store.resource_leases()[0]["status"], "unknown")
        finally:
            coordinator.stop()

    def test_resource_controller_credential_permissions_fail_closed(self) -> None:
        self.token.chmod(0o644)
        with self.assertRaises(AdmissionDeferred) as context:
            ResourceControlClient._token(self.token)
        self.assertEqual(context.exception.code, "controller_credential_invalid")

    def test_release_preserves_throttle_until_inference_profile_is_absent(self) -> None:
        control = FakeControlClient(active_profiles=["gpu.interactive"])
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        coordinator.start()
        try:
            handle = coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertIsNotNone(handle)
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.release(handle)
            self.assertEqual(context.exception.code, "release_unverified")
            self.assertEqual([call["mode"] for call in control.calls], ["interactive-boost"])
            self.assertTrue(coordinator.quarantined)
        finally:
            coordinator.stop()

    def test_execution_control_preserves_first_yield_reason(self) -> None:
        control = ExecutionControl(lambda: False)
        control.request_yield("interactive-arrived")
        control.request_yield("later-request")
        self.assertTrue(control.yield_requested())
        self.assertEqual(control.yield_reason, "interactive-arrived")

    def test_gpu_admission_requires_probe_even_with_default_policy(self) -> None:
        coordinator = ResourceCoordinator(store=self.store, data_root=self.root)
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_probe_required")
        finally:
            coordinator.stop()

    def test_stale_post_control_snapshot_rolls_back_without_admission(self) -> None:
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=StaleSnapshotControlClient(),
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_snapshot_stale")
            self.assertEqual(self.store.resource_leases()[0]["status"], "denied")
        finally:
            coordinator.stop()

    def test_controller_ack_requires_safe_boundary_and_causal_generation(self) -> None:
        controller = self.policy().controllers[0]
        response = {
            "leaseId": "lease_test", "fencingToken": "fence_test", "brokerEpoch": 4,
            "acknowledgedGeneration": 1, "effectiveMode": controller.throttled_mode,
            "health": "healthy", "snapshotGeneration": 9,
        }
        client = StubResourceControlClient(response)
        with self.assertRaises(AdmissionDeferred) as context:
            client.set_mode(
                controller, mode=controller.throttled_mode, lease_id="lease_test",
                fencing_token="fence_test", broker_epoch=4, generation=1, reason="test",
            )
        self.assertEqual(context.exception.code, "controller_unhealthy")

    def test_training_controller_requires_checkpoint_manifest_when_configured(self) -> None:
        controller = replace(self.policy().controllers[0], requires_checkpoint=True)
        response = {
            "leaseId": "lease_test", "fencingToken": "fence_test", "brokerEpoch": 4,
            "acknowledgedGeneration": 1, "effectiveMode": controller.throttled_mode,
            "health": "healthy", "appliedAtSafeBoundary": True, "snapshotGeneration": 9,
        }
        client = StubResourceControlClient(response)
        with self.assertRaises(AdmissionDeferred) as context:
            client.set_mode(
                controller, mode=controller.throttled_mode, lease_id="lease_test",
                fencing_token="fence_test", broker_epoch=4, generation=1, reason="test",
            )
        self.assertEqual(context.exception.code, "controller_checkpoint_missing")

    def test_restart_uses_new_epoch_takeover_after_profile_is_absent(self) -> None:
        control = FakeControlClient()
        first = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        first.start()
        first_epoch = first.epoch
        first.acquire(self.job, self.plan(), lambda: False)
        first.stop()
        second = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        second.start()
        try:
            self.assertGreater(second.epoch, first_epoch)
            self.assertFalse(second.quarantined)
            self.assertEqual(self.store.resource_leases()[0]["status"], "released")
            takeover = [call for call in control.calls if "recoveryFencingToken" in call]
            self.assertEqual(len(takeover), 1)
            self.assertEqual(takeover[0]["brokerEpoch"], second.epoch)
        finally:
            second.stop()

    def test_restart_allows_unthrottled_permit_to_remain_resident(self) -> None:
        policy = replace(self.policy(), controllers=())
        control = FakeControlClient(active_profiles=["gpu.interactive"])
        permit = replace(self.plan(), lease_mode="permit")
        first = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=policy,
            host_probe=FakeHostProbe(), control_client=control,
        )
        first.start()
        first.acquire(self.job, permit, lambda: False)
        first.stop()
        second = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=policy,
            host_probe=FakeHostProbe(), control_client=control,
        )
        second.start()
        try:
            self.assertFalse(second.quarantined)
            self.assertEqual(self.store.resource_leases()[0]["status"], "released")
        finally:
            second.stop()


if __name__ == "__main__":
    unittest.main()
