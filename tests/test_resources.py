from __future__ import annotations

import os
import tempfile
import threading
import time
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
        self.generation = 6
        self.controller_states: dict[str, dict] = {}

    def snapshot(self, endpoint, token_file):
        del endpoint, token_file
        self.generation += 1
        return {
            "health": "healthy", "unknownConsumers": self.unknown_consumers,
            "activeProfiles": self.active_profiles,
            "profiles": {
                profile_id: {
                    "health": "healthy",
                    "identityVerified": True,
                    "runtimeIdentity": f"test-runtime:{profile_id}",
                    "ownerId": "test.governor",
                }
                for profile_id in self.active_profiles
            },
            "controllerStates": self.controller_states,
            "generation": self.generation,
        }

    def set_mode(
        self, controller, *, mutation_id, mode, lease_id, fencing_token,
        broker_epoch, generation, reason,
    ):
        call = {
            "controller": controller.id, "mode": mode, "leaseId": lease_id,
            "fencingToken": fencing_token, "brokerEpoch": broker_epoch,
            "generation": generation, "reason": reason, "mutationId": mutation_id,
        }
        self.calls.append(call)
        if mode == controller.normal_mode:
            if controller.profile_id not in self.active_profiles:
                self.active_profiles.append(controller.profile_id)
        elif mode == controller.throttled_mode and controller.profile_id in self.active_profiles:
            self.active_profiles.remove(controller.profile_id)
        self.generation += 1
        acknowledgement = {
            "mutationId": mutation_id,
            "leaseId": lease_id,
            "fencingToken": fencing_token,
            "brokerEpoch": broker_epoch,
            "acknowledgedGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        self.controller_states[controller.id] = {
            "protocolVersion": "1.0",
            "controllerId": controller.id,
            "mutationId": mutation_id,
            "leaseId": lease_id,
            "fencingToken": fencing_token,
            "brokerEpoch": broker_epoch,
            "controlGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        return acknowledgement
    def takeover_mode(
        self, controller, *, mutation_id, previous_lease_id, previous_fencing_token,
        recovery_fencing_token, broker_epoch, generation, mode,
    ):
        self.calls.append({
            "controller": controller.id, "mode": mode,
            "previousLeaseId": previous_lease_id,
            "previousFencingToken": previous_fencing_token,
            "recoveryFencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "generation": generation,
            "mutationId": mutation_id,
        })
        if mode == controller.normal_mode and controller.profile_id not in self.active_profiles:
            self.active_profiles.append(controller.profile_id)
        self.generation += 1
        acknowledgement = {
            "mutationId": mutation_id,
            "previousLeaseId": previous_lease_id,
            "previousFencingToken": previous_fencing_token,
            "recoveryFencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "acknowledgedGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        self.controller_states[controller.id] = {
            "protocolVersion": "1.0",
            "controllerId": controller.id,
            "mutationId": mutation_id,
            "leaseId": previous_lease_id,
            "fencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "controlGeneration": generation,
            "effectiveMode": mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        return acknowledgement


class CheckpointingControlClient(FakeControlClient):
    def set_mode(self, controller, **kwargs):
        acknowledgement = super().set_mode(controller, **kwargs)
        if controller.requires_checkpoint and kwargs["mode"] == controller.throttled_mode:
            checkpoint = {
                "runId": "run_test",
                "checkpointId": "checkpoint_test",
                "sha256": "a" * 64,
            }
            acknowledgement["checkpoint"] = checkpoint
            self.controller_states[controller.id]["checkpoint"] = checkpoint
        return acknowledgement


class ConcurrentSnapshotControlClient(FakeControlClient):
    def __init__(self) -> None:
        super().__init__()
        self.active_calls = 0
        self.maximum_active_calls = 0
        self._calls_lock = threading.Lock()

    def snapshot(self, endpoint, token_file):
        with self._calls_lock:
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            time.sleep(0.02)
            return super().snapshot(endpoint, token_file)
        finally:
            with self._calls_lock:
                self.active_calls -= 1

class FailingRestoreControlClient(FakeControlClient):
    def set_mode(self, controller, *, mutation_id, mode, lease_id, fencing_token, broker_epoch, generation, reason):
        if mode == controller.normal_mode:
            raise AdmissionDeferred("restore_rejected", "controller rejected restore")
        return super().set_mode(
            controller, mutation_id=mutation_id, mode=mode, lease_id=lease_id, fencing_token=fencing_token,
            broker_epoch=broker_epoch, generation=generation, reason=reason,
        )


class JournalCheckingControlClient(FakeControlClient):
    def __init__(self, store: Store) -> None:
        super().__init__()
        self.store = store

    def set_mode(self, controller, *, mutation_id, mode, lease_id, fencing_token, broker_epoch, generation, reason):
        if mode == controller.throttled_mode:
            lease = self.store.get_resource_lease(lease_id)
            assert lease is not None
            assert lease["throttle"][-1]["state"] == "requested"
            assert lease["throttle"][-1]["controllerId"] == controller.id
        return super().set_mode(
            controller, mutation_id=mutation_id, mode=mode, lease_id=lease_id, fencing_token=fencing_token,
            broker_epoch=broker_epoch, generation=generation, reason=reason,
        )


class StaleSnapshotControlClient(FakeControlClient):
    def set_mode(self, controller, *args, **kwargs):
        value = super().set_mode(controller, *args, **kwargs)
        if kwargs.get("mode") == controller.throttled_mode:
            self.controller_states.pop(controller.id, None)
        return value


class NonAdvancingAckControlClient(FakeControlClient):
    def set_mode(self, controller, *args, **kwargs):
        value = super().set_mode(controller, *args, **kwargs)
        if kwargs.get("mode") == controller.throttled_mode:
            self.generation -= 2
        return value


class StaleRestoreControlClient(FakeControlClient):
    def set_mode(self, controller, *args, **kwargs):
        value = super().set_mode(controller, *args, **kwargs)
        if kwargs.get("mode") == controller.normal_mode:
            self.controller_states.pop(controller.id, None)
        return value


class StubResourceControlClient(ResourceControlClient):
    def __init__(self, response: dict) -> None:
        super().__init__()
        self.response = response

    def _request(self, method, url, token_file, payload, *, timeout=None):
        del method, url, token_file, payload, timeout
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

    def test_training_quantum_defers_repeated_displacement_before_mutation(self) -> None:
        controller = replace(
            self.policy().controllers[0],
            workload_kind="training",
            requires_checkpoint=True,
            minimum_normal_seconds=60,
        )
        policy = replace(
            self.policy(),
            controllers=(controller,),
            maximum_inference_window_seconds=30,
        )
        client = CheckpointingControlClient()
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=policy,
            host_probe=FakeHostProbe(), control_client=client,
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "training_quantum_active")
            self.assertGreater(context.exception.retry_after_seconds, 59)
            self.assertEqual(client.calls, [])
            self.assertEqual(self.store.resource_leases(), [])
        finally:
            coordinator.stop()

    def test_training_displacement_has_a_bounded_execution_window(self) -> None:
        controller = replace(
            self.policy().controllers[0],
            workload_kind="training",
            requires_checkpoint=True,
            minimum_normal_seconds=1,
        )
        policy = replace(
            self.policy(), controllers=(controller,), maximum_inference_window_seconds=20,
        )
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=policy,
            host_probe=FakeHostProbe(), control_client=FakeControlClient(),
        )
        control = ExecutionControl(lambda: False)
        coordinator.configure_execution_control(self.plan(), control)
        self.assertLessEqual(control.bounded_timeout(600), 20)
        self.assertFalse(control())

    def test_training_policy_requires_quantum_and_window(self) -> None:
        base = {
            "version": 1,
            "controllers": [{
                "id": "trainer", "profileId": "gpu.trainer", "workloadKind": "training",
                "endpoint": "http://127.0.0.1:9001", "tokenFile": "/tmp/test-token",
                "throttleFor": ["interactive"], "requiresCheckpoint": True,
            }],
        }
        with self.assertRaisesRegex(ValueError, "minimumNormalSeconds"):
            ResourcePolicy.from_value(base)
        base["controllers"][0]["minimumNormalSeconds"] = 60
        with self.assertRaisesRegex(ValueError, "maximumInferenceWindowSeconds"):
            ResourcePolicy.from_value(base)

    def test_gpu_admission_requires_probe_even_with_default_policy(self) -> None:
        coordinator = ResourceCoordinator(store=self.store, data_root=self.root)
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_probe_required")
        finally:
            coordinator.stop()

    def test_unobserved_controller_mutation_rolls_back_without_admission(self) -> None:
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=StaleSnapshotControlClient(),
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "controller_observation_missing")
            self.assertEqual(self.store.resource_leases()[0]["status"], "denied")
        finally:
            coordinator.stop()

    def test_controller_mutation_requires_a_newer_probe_observation(self) -> None:
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=NonAdvancingAckControlClient(),
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "controller_observation_stale")
            self.assertEqual(self.store.resource_leases()[0]["status"], "denied")
        finally:
            coordinator.stop()

    def test_controller_ack_requires_safe_boundary(self) -> None:
        controller = self.policy().controllers[0]
        mutation_id = "mutation_" + "a" * 32
        response = {
            "mutationId": mutation_id,
            "leaseId": "lease_test", "fencingToken": "fence_test", "brokerEpoch": 4,
            "acknowledgedGeneration": 1, "effectiveMode": controller.throttled_mode,
            "health": "healthy",
        }
        client = StubResourceControlClient(response)
        with self.assertRaises(AdmissionDeferred) as context:
            client.set_mode(
                controller, mutation_id=mutation_id, mode=controller.throttled_mode, lease_id="lease_test",
                fencing_token="fence_test", broker_epoch=4, generation=1, reason="test",
            )
        self.assertEqual(context.exception.code, "controller_unhealthy")

    def test_training_controller_requires_checkpoint_manifest_when_configured(self) -> None:
        controller = replace(self.policy().controllers[0], requires_checkpoint=True)
        mutation_id = "mutation_" + "b" * 32
        response = {
            "mutationId": mutation_id,
            "leaseId": "lease_test", "fencingToken": "fence_test", "brokerEpoch": 4,
            "acknowledgedGeneration": 1, "effectiveMode": controller.throttled_mode,
            "health": "healthy", "appliedAtSafeBoundary": True,
        }
        client = StubResourceControlClient(response)
        with self.assertRaises(AdmissionDeferred) as context:
            client.set_mode(
                controller, mutation_id=mutation_id, mode=controller.throttled_mode, lease_id="lease_test",
                fencing_token="fence_test", broker_epoch=4, generation=1, reason="test",
            )
        self.assertEqual(context.exception.code, "controller_checkpoint_missing")

    def test_takeover_requires_mutation_identity_echo(self) -> None:
        controller = self.policy().controllers[0]
        mutation_id = "mutation_" + "c" * 32
        response = {
            "previousLeaseId": "lease_previous",
            "previousFencingToken": "fence_previous",
            "recoveryFencingToken": "fence_recovery",
            "brokerEpoch": 5,
            "acknowledgedGeneration": 1,
            "effectiveMode": controller.normal_mode,
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        client = StubResourceControlClient(response)
        with self.assertRaises(AdmissionDeferred) as context:
            client.takeover_mode(
                controller,
                mutation_id=mutation_id,
                previous_lease_id="lease_previous",
                previous_fencing_token="fence_previous",
                recovery_fencing_token="fence_recovery",
                broker_epoch=5,
                generation=1,
                mode=controller.normal_mode,
            )
        self.assertEqual(context.exception.code, "controller_takeover_mismatch")

    def test_resource_probe_rejects_boolean_generation(self) -> None:
        control = FakeControlClient()
        original_snapshot = control.snapshot

        def invalid_snapshot(endpoint, token_file):
            value = original_snapshot(endpoint, token_file)
            value["generation"] = True
            return value

        control.snapshot = invalid_snapshot
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            policy=self.policy(),
            host_probe=FakeHostProbe(),
            control_client=control,
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_probe_invalid")
        finally:
            coordinator.stop()

    def test_resource_probe_rejects_omitted_inventory_fields(self) -> None:
        control = FakeControlClient()
        control.snapshot = lambda endpoint, token_file: {"health": "healthy", "generation": 1}
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            policy=self.policy(),
            host_probe=FakeHostProbe(),
            control_client=control,
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_probe_invalid")
        finally:
            coordinator.stop()

    def test_resource_probe_rejects_unverified_active_profile_identity(self) -> None:
        control = FakeControlClient(active_profiles=["gpu.interactive"])
        original_snapshot = control.snapshot

        def unverified_snapshot(endpoint, token_file):
            value = original_snapshot(endpoint, token_file)
            value["profiles"]["gpu.interactive"].pop("identityVerified")
            return value

        control.snapshot = unverified_snapshot
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            policy=self.policy(),
            host_probe=FakeHostProbe(),
            control_client=control,
        )
        coordinator.start()
        try:
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.acquire(self.job, self.plan(), lambda: False)
            self.assertEqual(context.exception.code, "resource_probe_invalid")
        finally:
            coordinator.stop()

    def test_resource_probe_rejects_generation_regression(self) -> None:
        control = FakeControlClient()
        generations = iter((7, 6))
        original_snapshot = control.snapshot

        def regressing_snapshot(endpoint, token_file):
            value = original_snapshot(endpoint, token_file)
            value["generation"] = next(generations)
            return value

        control.snapshot = regressing_snapshot
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            policy=self.policy(),
            host_probe=FakeHostProbe(),
            control_client=control,
        )
        coordinator.start()
        try:
            coordinator._combined_snapshot()
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator._combined_snapshot()
            self.assertEqual(context.exception.code, "resource_probe_regressed")
        finally:
            coordinator.stop()

    def test_concurrent_status_samples_are_serialized_before_generation_validation(self) -> None:
        control = ConcurrentSnapshotControlClient()
        coordinator = ResourceCoordinator(
            store=self.store, data_root=self.root, policy=self.policy(),
            host_probe=FakeHostProbe(), control_client=control,
        )
        coordinator.start()
        try:
            threads = [threading.Thread(target=coordinator.routing_context) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(2)
                self.assertFalse(thread.is_alive())
        finally:
            coordinator.stop()
        self.assertEqual(control.maximum_active_calls, 1)

    def test_release_waits_for_causal_controller_restoration(self) -> None:
        control = StaleRestoreControlClient()
        coordinator = ResourceCoordinator(
            store=self.store,
            data_root=self.root,
            policy=self.policy(),
            host_probe=FakeHostProbe(),
            control_client=control,
        )
        coordinator.start()
        try:
            handle = coordinator.acquire(self.job, self.plan(), lambda: False)
            with self.assertRaises(AdmissionDeferred) as context:
                coordinator.release(handle)
            self.assertEqual(context.exception.code, "controller_observation_missing")
            self.assertTrue(coordinator.quarantined)
            self.assertEqual(self.store.resource_leases()[0]["status"], "unknown")
        finally:
            coordinator.stop()

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
