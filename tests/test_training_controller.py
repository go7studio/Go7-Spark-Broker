from __future__ import annotations

import hashlib
import http.client
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from spark_broker.resource_probe import ControllerStateFile, load_controller_state
from spark_broker.training_controller import (
    TrainingController,
    TrainingControllerError,
    TrainingControllerHTTPServer,
    TrainingControllerPolicy,
    TrainingControllerServerConfig,
)


class FakeUnits:
    def __init__(self, *, active: bool = True, on_stop=None) -> None:
        self.is_active = active
        self.on_stop = on_stop
        self.starts = 0
        self.stops = 0

    def start(self, unit: str, timeout: int) -> None:
        self.starts += 1
        self.is_active = True

    def stop(self, unit: str, timeout: int) -> None:
        self.stops += 1
        self.is_active = False
        if self.on_stop is not None:
            self.on_stop()

    def active(self, unit: str) -> bool:
        return self.is_active


class TrainingControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir(mode=0o700)
        self.receipt = self.checkpoints / "latest.json"
        self.state = self.root / "state" / "trainer.json"
        self.authority = self.root / "authority" / "trainer.json"
        self.policy = TrainingControllerPolicy(
            controller_id="trainer",
            profile_id="bloom-v40",
            unit="bloom-v40.service",
            normal_mode="running",
            released_mode="released",
            checkpoint_root=self.checkpoints.resolve(),
            checkpoint_receipt_file=self.receipt,
            state_file=self.state,
            authority_file=self.authority,
            allow_fresh_start=False,
            stop_timeout_seconds=30,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def request(
        *,
        generation: int,
        mode: str,
        mutation: str = "1",
        lease: str = "1",
        fence: str = "one",
        epoch: int = 1,
    ) -> dict[str, object]:
        return {
            "protocolVersion": "1.0",
            "mutationId": f"mutation_{mutation * 32}",
            "leaseId": f"lease_{lease * 32}",
            "fencingToken": f"fence_{fence}",
            "brokerEpoch": epoch,
            "controlGeneration": generation,
            "targetMode": mode,
            "reason": "test transition",
        }

    @staticmethod
    def takeover_request() -> dict[str, object]:
        return {
            "protocolVersion": "1.0",
            "mutationId": f"mutation_{'3' * 32}",
            "previousLeaseId": f"lease_{'1' * 32}",
            "previousFencingToken": "fence_one",
            "recoveryFencingToken": "fence_recovery",
            "brokerEpoch": 2,
            "controlGeneration": 2,
            "targetMode": "running",
            "reason": "restart recovery",
        }

    def write_checkpoint(self, checkpoint_id: str, content: bytes = b"checkpoint") -> None:
        artifact = self.checkpoints / "model.bin"
        artifact.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        value = {
            "version": 1,
            "runId": "run-1",
            "checkpointId": checkpoint_id,
            "files": [{"path": "model.bin", "size": len(content), "sha256": digest}],
        }
        self.receipt.write_text(json.dumps(value), encoding="utf-8")
        self.receipt.chmod(0o600)

    def test_release_requires_and_publishes_verified_checkpoint(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)

        response = controller.set_mode(self.request(generation=1, mode="released"))

        self.assertEqual(response["effectiveMode"], "released")
        self.assertEqual(response["checkpoint"]["checkpointId"], "checkpoint-1")
        self.assertEqual(units.stops, 1)
        observed = load_controller_state(ControllerStateFile("trainer", self.state))
        self.assertEqual(observed["checkpoint"], response["checkpoint"])

    def test_resume_revalidates_checkpoint_before_start(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released"))

        response = controller.set_mode(self.request(generation=2, mode="running", mutation="2"))

        self.assertEqual(response["effectiveMode"], "running")
        self.assertEqual(response["checkpoint"]["checkpointId"], "checkpoint-1")
        self.assertEqual(units.starts, 1)
        self.assertTrue(units.is_active)

    def test_resume_refuses_changed_checkpoint(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released"))
        (self.checkpoints / "model.bin").write_bytes(b"tampered")

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=2, mode="running", mutation="2"))

        self.assertEqual(context.exception.code, "checkpoint_changed")
        self.assertEqual(units.starts, 0)

    def test_each_new_release_requires_an_advanced_checkpoint(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released"))
        controller.set_mode(self.request(generation=2, mode="running", mutation="2"))

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=3, mode="released", mutation="3"))

        self.assertEqual(context.exception.code, "checkpoint_not_advanced")
        self.assertFalse(units.is_active)

    def test_applying_transition_is_recovered_on_restart(self) -> None:
        request = self.request(generation=1, mode="released")
        self.authority.parent.mkdir(mode=0o700)
        self.authority.write_text(
            json.dumps({
                "version": 1,
                "phase": "applying",
                "request": request,
                "previous": None,
                "releaseBaseline": None,
            }),
            encoding="utf-8",
        )
        self.authority.chmod(0o600)
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))

        controller = TrainingController(self.policy, units=units)
        response = controller.set_mode(request)

        self.assertEqual(response["checkpoint"]["checkpointId"], "checkpoint-1")
        self.assertEqual(units.stops, 1)
        persisted = json.loads(self.authority.read_text(encoding="utf-8"))
        self.assertEqual(persisted["phase"], "ready")

    def test_applying_release_recovery_replays_captured_baseline(self) -> None:
        self.write_checkpoint("checkpoint-before-stop")
        request = self.request(generation=1, mode="released")
        baseline = TrainingController(
            self.policy, units=FakeUnits()
        )._receipt_identity_if_present()
        self.assertIsNotNone(baseline)
        self.authority.parent.mkdir(mode=0o700)
        self.authority.write_text(
            json.dumps({
                "version": 1,
                "phase": "applying",
                "request": request,
                "previous": None,
                "releaseBaseline": baseline,
            }),
            encoding="utf-8",
        )
        self.authority.chmod(0o600)
        units = FakeUnits(
            on_stop=lambda: self.write_checkpoint(
                "checkpoint-after-stop", b"new durable training state"
            )
        )

        response = TrainingController(self.policy, units=units).set_mode(request)

        self.assertEqual(response["checkpoint"]["checkpointId"], "checkpoint-after-stop")
        self.assertEqual(units.stops, 1)
        persisted = json.loads(self.authority.read_text(encoding="utf-8"))
        self.assertEqual(persisted["phase"], "ready")

    def test_applying_release_recovery_rejects_stale_captured_baseline(self) -> None:
        self.write_checkpoint("checkpoint-before-stop")
        request = self.request(generation=1, mode="released")
        baseline = TrainingController(
            self.policy, units=FakeUnits()
        )._receipt_identity_if_present()
        self.assertIsNotNone(baseline)
        self.authority.parent.mkdir(mode=0o700)
        self.authority.write_text(
            json.dumps({
                "version": 1,
                "phase": "applying",
                "request": request,
                "previous": None,
                "releaseBaseline": baseline,
            }),
            encoding="utf-8",
        )
        self.authority.chmod(0o600)
        units = FakeUnits()

        with self.assertRaises(TrainingControllerError) as context:
            TrainingController(self.policy, units=units).set_mode(request)

        self.assertEqual(context.exception.code, "checkpoint_not_advanced")
        self.assertEqual(units.stops, 1)
        persisted = json.loads(self.authority.read_text(encoding="utf-8"))
        self.assertEqual(persisted["phase"], "applying")
        self.assertEqual(persisted["releaseBaseline"], baseline)

    def test_first_release_does_not_accept_a_preexisting_stale_receipt(self) -> None:
        self.write_checkpoint("checkpoint-before-stop")
        units = FakeUnits()
        controller = TrainingController(self.policy, units=units)

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=1, mode="released"))

        self.assertEqual(context.exception.code, "checkpoint_not_advanced")
        self.assertEqual(units.stops, 1)

    def test_release_rejects_id_only_bump_with_unchanged_manifest(self) -> None:
        self.write_checkpoint("checkpoint-before-stop")
        units = FakeUnits(
            on_stop=lambda: self.write_checkpoint("checkpoint-after-stop")
        )
        controller = TrainingController(self.policy, units=units)

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=1, mode="released"))

        self.assertEqual(context.exception.code, "checkpoint_not_advanced")

    def test_release_accepts_new_same_run_manifest_evidence(self) -> None:
        self.write_checkpoint("checkpoint-before-stop")
        units = FakeUnits(
            on_stop=lambda: self.write_checkpoint(
                "checkpoint-after-stop", b"new durable training state"
            )
        )
        controller = TrainingController(self.policy, units=units)

        response = controller.set_mode(self.request(generation=1, mode="released"))

        self.assertEqual(response["checkpoint"]["checkpointId"], "checkpoint-after-stop")

    def test_release_baseline_is_the_latest_receipt_not_prior_controller_state(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released"))
        controller.set_mode(self.request(generation=2, mode="running", mutation="2"))
        self.write_checkpoint("checkpoint-periodic")
        units.on_stop = None

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=3, mode="released", mutation="3"))

        self.assertEqual(context.exception.code, "checkpoint_not_advanced")

    def test_takeover_restores_prior_checkpoint_under_new_fence(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released"))

        response = controller.takeover(self.takeover_request())

        self.assertEqual(response["recoveryFencingToken"], "fence_recovery")
        self.assertEqual(response["effectiveMode"], "running")
        self.assertTrue(units.is_active)
        observed = load_controller_state(ControllerStateFile("trainer", self.state))
        self.assertEqual(observed["fencingToken"], "fence_recovery")
        self.assertEqual(observed["brokerEpoch"], 2)

    def test_stale_epoch_is_rejected(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        controller.set_mode(self.request(generation=1, mode="released", epoch=2))

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(
                self.request(generation=2, mode="running", mutation="2", epoch=1)
            )

        self.assertEqual(context.exception.code, "stale_epoch")

    def test_fresh_training_start_is_disabled_by_default(self) -> None:
        units = FakeUnits(active=False)
        controller = TrainingController(self.policy, units=units)

        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=1, mode="running"))

        self.assertEqual(context.exception.code, "checkpoint_required")
        self.assertEqual(units.starts, 0)

    def test_checkpoint_symlink_is_rejected(self) -> None:
        outside = self.root / "outside.bin"
        outside.write_bytes(b"outside")

        def publish() -> None:
            (self.checkpoints / "model.bin").symlink_to(outside)
            digest = hashlib.sha256(b"outside").hexdigest()
            self.receipt.write_text(
                json.dumps({
                    "version": 1,
                    "runId": "run-1",
                    "checkpointId": "checkpoint-1",
                    "files": [{"path": "model.bin", "size": 7, "sha256": digest}],
                }),
                encoding="utf-8",
            )
            self.receipt.chmod(0o600)

        controller = TrainingController(self.policy, units=FakeUnits(on_stop=publish))
        with self.assertRaises(TrainingControllerError) as context:
            controller.set_mode(self.request(generation=1, mode="released"))
        self.assertEqual(context.exception.code, "checkpoint_invalid")

    def test_policy_rejects_identical_modes(self) -> None:
        config = self.root / "controller.json"
        config.write_text(json.dumps({
            "version": 1,
            "controllerId": "trainer",
            "profileId": "bloom-v40",
            "unit": "bloom-v40.service",
            "normalMode": "running",
            "releasedMode": "running",
            "checkpointRoot": str(self.checkpoints),
            "checkpointReceiptFile": str(self.receipt),
            "stateFile": str(self.state),
            "authorityFile": str(self.authority),
        }), encoding="utf-8")
        config.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "must differ"):
            TrainingControllerPolicy.from_file(config)

    def test_http_requires_bearer_and_serves_typed_transition(self) -> None:
        units = FakeUnits(on_stop=lambda: self.write_checkpoint("checkpoint-1"))
        controller = TrainingController(self.policy, units=units)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
        reservation.close()
        server = TrainingControllerHTTPServer(
            TrainingControllerServerConfig("127.0.0.1", port, "t" * 32),
            controller,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = json.dumps(self.request(generation=1, mode="released"))
            connection.request(
                "POST",
                "/v1/resource-mode",
                body=body,
                headers={"Authorization": f"Bearer {'t' * 32}", "Content-Type": "application/json"},
            )
            response = connection.getresponse()
            value = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(value["checkpoint"]["checkpointId"], "checkpoint-1")
            connection.close()

            denied = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            denied.request("POST", "/v1/resource-mode", body=body)
            denied_response = denied.getresponse()
            denied_value = json.loads(denied_response.read())
            self.assertEqual(denied_response.status, 401)
            self.assertEqual(denied_value["error"]["code"], "unauthorized")
            denied.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
