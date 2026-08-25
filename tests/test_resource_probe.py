from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from spark_broker.resource_probe import (
    GenerationStore,
    ProbeConfigError,
    ProbeHTTPServer,
    ProbePolicy,
    ProbeServerConfig,
    ResourceInventory,
    load_policy,
    load_token,
)


DOCKER_ID = "d" * 64
DOCKER_IMAGE = "sha256:" + "a" * 64
SYSTEMD_EXE = "sha256:" + "b" * 64
PMON = ("nvidia-smi", "pmon", "-c", "1", "-s", "m")
GPU_MEMORY_USED = ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")


def pmon(*pids: int) -> str:
    rows = ["# gpu pid type fb ccpm command"]
    rows.extend(f"0 {pid} C 1 0 fixture" for pid in pids)
    if not pids:
        rows.append("0 - - - - -")
    return "\n".join(rows) + "\n"


def policy_bytes(*profiles: dict, controller_state_files: tuple[dict, ...] = ()) -> bytes:
    value = {
        "version": 1,
        "ownerId": "spark.probe",
        "profiles": list(profiles),
    }
    if controller_state_files:
        value["controllerStateFiles"] = list(controller_state_files)
    return json.dumps(value).encode("utf-8")


class FixtureCommands:
    def __init__(self, values: dict[tuple[str, ...], str | Exception]) -> None:
        self.values = values
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...]) -> str:
        self.calls.append(argv)
        value = self.values[argv]
        if isinstance(value, Exception):
            raise value
        return value


class FixtureProcesses:
    def __init__(self, cgroups: dict[int, str], executables: dict[int, str] | None = None) -> None:
        self.cgroups = cgroups
        self.executables = executables or {}

    def cgroup(self, pid: int) -> str:
        value = self.cgroups[pid]
        if value == "ERROR":
            raise OSError("vanished")
        return value

    def executable_sha256(self, pid: int) -> str:
        return self.executables[pid]


class ResourceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.meminfo = self.root / "meminfo"
        self.meminfo.write_text(
            "MemTotal:       131072000 kB\nMemAvailable:   104857600 kB\nSwapFree:               0 kB\n",
            encoding="utf-8",
        )
        self.pressure = self.root / "pressure"
        self.pressure.write_text(
            "some avg10=0.25 avg60=0.10 avg300=0.05 total=10\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
            encoding="utf-8",
        )
        self.generation = GenerationStore(self.root / "generation")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def inventory(
        self,
        policy: ProbePolicy,
        commands: FixtureCommands,
        processes: FixtureProcesses,
    ) -> ResourceInventory:
        return ResourceInventory(
            policy,
            self.generation,
            command=commands,
            processes=processes,
            meminfo_path=self.meminfo,
            pressure_path=self.pressure,
            monotonic=lambda: 12.5,
        )

    def test_docker_consumer_is_bound_to_pinned_image_and_cgroup(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes({
            "id": "gpu.text",
            "type": "docker",
            "container": "installed-text-runtime",
            "expectedIdentity": DOCKER_IMAGE,
        }))
        commands = FixtureCommands({
            ("docker", "inspect", "installed-text-runtime"): json.dumps([{
                "Id": DOCKER_ID, "Image": DOCKER_IMAGE, "State": {"Running": True},
            }]),
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "4242, 4096\n",
            PMON: pmon(4242),
        })
        processes = FixtureProcesses({4242: f"0::/system.slice/docker-{DOCKER_ID}.scope\n"})

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "healthy")
        self.assertEqual(value["unknownConsumers"], 0)
        self.assertEqual(value["activeProfiles"], ["gpu.text"])
        self.assertEqual(value["profiles"]["gpu.text"], {
            "health": "healthy",
            "identityVerified": True,
            "runtimeIdentity": f"oci-image:{DOCKER_IMAGE};container:{DOCKER_ID}",
            "ownerId": "spark.probe",
            "gpuMemoryBytes": 4096 * 1024 * 1024,
        })
        self.assertEqual(value["metrics"], {
            "umaTotalBytes": 131072000 * 1024,
            "umaAvailableBytes": 104857600 * 1024,
            "swapFreeBytes": 0,
            "memoryPressureSomeAvg10": 0.25,
            "sampledAtMonotonic": 12.5,
        })
        self.assertEqual(value["observabilityErrors"], [])

    def test_docker_cgroup_accepts_only_full_or_canonical_short_identity(self) -> None:
        profile = {
            "id": "gpu.text", "type": "docker", "container": "installed-text-runtime",
            "expectedIdentity": DOCKER_IMAGE,
        }
        inspect = json.dumps([{
            "Id": DOCKER_ID, "Image": DOCKER_IMAGE, "State": {"Running": True},
        }])
        command_values = {
            ("docker", "inspect", "installed-text-runtime"): inspect,
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "4242, 1\n",
            PMON: pmon(4242),
        }
        short = self.inventory(
            ProbePolicy.from_bytes(policy_bytes(profile)), FixtureCommands(command_values),
            FixtureProcesses({4242: f"0::/docker-{DOCKER_ID[:12]}.scope\n"}),
        ).snapshot()
        self.assertEqual(short["activeProfiles"], ["gpu.text"])

        partial = self.inventory(
            ProbePolicy.from_bytes(policy_bytes(profile)), FixtureCommands(command_values),
            FixtureProcesses({4242: f"0::/docker-{DOCKER_ID[:13]}.scope\n"}),
        ).snapshot()
        self.assertEqual(partial["activeProfiles"], [])
        self.assertEqual(partial["unknownConsumers"], 1)

    def test_systemd_child_consumer_is_bound_to_unit_cgroup_and_executable(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes({
            "id": "gpu.training",
            "type": "systemd-user",
            "unit": "installed-training.service",
            "expectedIdentity": SYSTEMD_EXE,
        }))
        systemctl = (
            "ActiveState=active\nSubState=running\nMainPID=100\n"
            "ControlGroup=/user.slice/user-1000.slice/user@1000.service/app.slice/installed-training.service\n"
        )
        commands = FixtureCommands({
            (
                "systemctl", "--user", "show", "installed-training.service",
                "--property=ActiveState", "--property=SubState", "--property=MainPID", "--property=ControlGroup",
            ): systemctl,
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "101, 8192\n",
            PMON: pmon(101),
        })
        cgroup = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/installed-training.service/workers\n"
        processes = FixtureProcesses({101: cgroup}, {100: SYSTEMD_EXE, 101: SYSTEMD_EXE})

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "healthy")
        self.assertEqual(value["activeProfiles"], ["gpu.training"])
        self.assertEqual(value["unknownConsumers"], 0)
        self.assertTrue(value["profiles"]["gpu.training"]["runtimeIdentity"].startswith("systemd-exe:sha256:"))

    def test_systemd_gpu_child_must_match_the_pinned_executable(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes({
            "id": "gpu.training",
            "type": "systemd-user",
            "unit": "installed-training.service",
            "expectedIdentity": SYSTEMD_EXE,
        }))
        systemctl = (
            "ActiveState=active\nSubState=running\nMainPID=100\n"
            "ControlGroup=/user.slice/installed-training.service\n"
        )
        commands = FixtureCommands({
            (
                "systemctl", "--user", "show", "installed-training.service",
                "--property=ActiveState", "--property=SubState", "--property=MainPID", "--property=ControlGroup",
            ): systemctl,
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "101, 8192\n",
            PMON: pmon(101),
        })
        processes = FixtureProcesses(
            {101: "0::/user.slice/installed-training.service/workers\n"},
            {100: SYSTEMD_EXE, 101: "sha256:" + "c" * 64},
        )

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 1)
        self.assertEqual(value["activeProfiles"], [])
        self.assertIn("consumer_identity_mismatch:101", value["observabilityErrors"])

    def test_unknown_and_vanished_consumers_fail_closed(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes())
        commands = FixtureCommands({
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "201, 1\n202, 2\n",
            PMON: pmon(201, 202),
        })
        processes = FixtureProcesses({201: "0::/unknown\n", 202: "ERROR"})

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 2)
        self.assertEqual(value["activeProfiles"], [])
        self.assertIn("consumer_unknown:201", value["observabilityErrors"])
        self.assertIn("consumer_unobservable:202", value["observabilityErrors"])

    def test_gpu_enumeration_failure_reports_synthetic_unknown(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes())
        commands = FixtureCommands({
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): RuntimeError("offline"),
            PMON: pmon(),
        })

        value = self.inventory(policy, commands, FixtureProcesses({})).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 1)
        self.assertIn("gpu_inventory_unobservable", value["observabilityErrors"])

    def test_graphics_only_process_from_pmon_is_not_hidden(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes())
        commands = FixtureCommands({
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "",
            PMON: "# gpu pid type fb ccpm command\n0 303 G 64 0 graphics\n",
        })

        value = self.inventory(
            policy, commands, FixtureProcesses({303: "0::/unmanaged-graphics\n"})
        ).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 1)
        self.assertIn("consumer_unknown:303", value["observabilityErrors"])

    def test_unattributed_reported_gpu_memory_creates_a_synthetic_unknown(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes({
            "id": "gpu.text",
            "type": "docker",
            "container": "installed-text-runtime",
            "expectedIdentity": DOCKER_IMAGE,
        }))
        commands = FixtureCommands({
            ("docker", "inspect", "installed-text-runtime"): json.dumps([{
                "Id": DOCKER_ID, "Image": DOCKER_IMAGE, "State": {"Running": True},
            }]),
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "4242, 128\n",
            PMON: pmon(4242),
            GPU_MEMORY_USED: "1024\n",
        })
        processes = FixtureProcesses({4242: f"0::/docker-{DOCKER_ID}.scope\n"})

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 1)
        self.assertFalse(value["gpuMemory"]["reconciled"])
        self.assertEqual(value["gpuMemory"]["residualBytes"], 896 * 1024 * 1024)
        self.assertIn("gpu_memory_residual_unattributed", value["observabilityErrors"])

    def test_runtime_identity_mismatch_is_degraded_and_consumer_unknown(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes({
            "id": "gpu.text",
            "type": "docker",
            "container": "installed-text-runtime",
            "expectedIdentity": "sha256:" + "c" * 64,
        }))
        commands = FixtureCommands({
            ("docker", "inspect", "installed-text-runtime"): json.dumps([{
                "Id": DOCKER_ID, "Image": DOCKER_IMAGE, "State": {"Running": True},
            }]),
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "4242, N/A\n",
            PMON: pmon(4242),
        })
        processes = FixtureProcesses({4242: f"0::/docker/{DOCKER_ID}\n"})

        value = self.inventory(policy, commands, processes).snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 1)
        self.assertEqual(value["profiles"], {})
        self.assertIn("identity_mismatch:gpu.text", value["observabilityErrors"])

    def test_missing_memory_or_psi_is_explicitly_degraded(self) -> None:
        policy = ProbePolicy.from_bytes(policy_bytes())
        commands = FixtureCommands({
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "",
            PMON: pmon(),
        })
        inventory = ResourceInventory(
            policy,
            self.generation,
            command=commands,
            processes=FixtureProcesses({}),
            meminfo_path=self.root / "missing-meminfo",
            pressure_path=self.root / "missing-pressure",
        )

        value = inventory.snapshot()

        self.assertEqual(value["health"], "degraded")
        self.assertEqual(value["unknownConsumers"], 0)
        self.assertIsNone(value["metrics"]["umaAvailableBytes"])
        self.assertIn("uma_memory_unobservable", value["observabilityErrors"])
        self.assertIn("memory_psi_unobservable", value["observabilityErrors"])

    def test_generation_is_durable_and_monotonic_across_instances(self) -> None:
        first = GenerationStore(self.root / "generation")
        second = GenerationStore(self.root / "generation")
        self.assertEqual(first.next(), 1)
        self.assertEqual(second.next(), 2)
        self.assertEqual((self.root / "generation").read_text(encoding="ascii"), "2\n")
        self.assertEqual((self.root / "generation").stat().st_mode & 0o777, 0o600)

    def test_controller_mutation_state_is_published_for_causal_observation(self) -> None:
        state_file = self.root / "training-controller-state.json"
        state = {
            "protocolVersion": "1.0",
            "controllerId": "background-training",
            "mutationId": "mutation_" + "a" * 32,
            "leaseId": "lease_" + "b" * 32,
            "fencingToken": "fence_7_" + "c" * 32,
            "brokerEpoch": 7,
            "controlGeneration": 1,
            "effectiveMode": "checkpoint-release",
            "health": "healthy",
            "appliedAtSafeBoundary": True,
            "checkpoint": {
                "runId": "run-test", "checkpointId": "checkpoint-test", "sha256": "d" * 64,
            },
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")
        state_file.chmod(0o600)
        policy = ProbePolicy.from_bytes(policy_bytes(
            controller_state_files=({
                "id": "background-training", "stateFile": str(state_file),
            },),
        ))
        commands = FixtureCommands({
            ("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"): "",
            PMON: pmon(),
        })
        value = self.inventory(policy, commands, FixtureProcesses({})).snapshot()
        self.assertEqual(value["health"], "healthy")
        self.assertEqual(value["controllerStates"], {"background-training": state})

        state_file.write_text("{}", encoding="utf-8")
        degraded = self.inventory(policy, commands, FixtureProcesses({})).snapshot()
        self.assertEqual(degraded["health"], "degraded")
        self.assertIn(
            "controller_state_unobservable:background-training",
            degraded["observabilityErrors"],
        )

    def test_policy_and_token_reject_symlinks_and_permissive_modes(self) -> None:
        config = self.root / "probe.json"
        config.write_bytes(policy_bytes())
        config.chmod(0o600)
        token = self.root / "token"
        token.write_text("t" * 32, encoding="utf-8")
        token.chmod(0o600)
        self.assertEqual(load_policy(config).owner_id, "spark.probe")
        self.assertEqual(load_token(token), "t" * 32)

        config.chmod(0o644)
        with self.assertRaisesRegex(ProbeConfigError, "mode 0600"):
            load_policy(config)
        config.chmod(0o600)
        link = self.root / "token-link"
        link.symlink_to(token)
        with self.assertRaisesRegex(ProbeConfigError, "opened safely"):
            load_token(link)

    def test_policy_schema_is_closed_and_requires_immutable_identity(self) -> None:
        with self.assertRaisesRegex(ProbeConfigError, "optional controllerStateFiles"):
            ProbePolicy.from_bytes(json.dumps({
                "version": 1, "ownerId": "spark.probe", "profiles": [], "extra": True,
            }).encode())
        with self.assertRaisesRegex(ProbeConfigError, "sha256"):
            ProbePolicy.from_bytes(policy_bytes({
                "id": "gpu.text", "type": "docker", "container": "runtime", "expectedIdentity": "latest",
            }))

    def test_http_endpoint_is_loopback_bearer_authenticated_and_no_store(self) -> None:
        class StaticInventory:
            def snapshot(self):
                return {
                    "health": "healthy", "generation": 1, "unknownConsumers": 0,
                    "activeProfiles": [], "profiles": {}, "metrics": {}, "observabilityErrors": [],
                }

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        server = ProbeHTTPServer(
            ProbeServerConfig("127.0.0.1", port, "s" * 32), StaticInventory(),  # type: ignore[arg-type]
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}/v1/resource-snapshot"
            with self.assertRaises(urllib.error.HTTPError) as unauthorized:
                urllib.request.urlopen(url, timeout=2)
            self.assertEqual(unauthorized.exception.code, 401)
            unauthorized.exception.close()

            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {'s' * 32}"})
            with urllib.request.urlopen(request, timeout=2) as response:
                value = json.load(response)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(value["generation"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_non_loopback_bind_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProbeConfigError, "loopback"):
            ProbeServerConfig("0.0.0.0", 8791, "x" * 32)


if __name__ == "__main__":
    unittest.main()
