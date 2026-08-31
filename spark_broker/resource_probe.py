from __future__ import annotations

import fcntl
import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol

from .secure_files import SecureFileError, read_owner_file


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SYSTEMD_UNIT = re.compile(r"^[A-Za-z0-9:_.@\\-]{1,240}\.service$")
_SHA256_IDENTITY = re.compile(r"^sha256:[a-f0-9]{64}$")
_CONTAINER_ID = re.compile(r"(?<![a-f0-9])([a-f0-9]{12,64})(?![a-f0-9])")
_MUTATION_ID = re.compile(r"^mutation_[a-f0-9]{32}$")
_LEASE_ID = re.compile(r"^lease_[a-f0-9]{32}$")
_FENCE = re.compile(r"^fence_[A-Za-z0-9_]{1,240}$")
_GPU_MEMORY_RECONCILIATION_TOLERANCE_BYTES = 64 * 1024 * 1024


class ProbeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class InstalledProfile:
    id: str
    kind: str
    expected_identity: str
    container: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class ControllerStateFile:
    id: str
    path: Path


@dataclass(frozen=True)
class ProbePolicy:
    owner_id: str
    profiles: tuple[InstalledProfile, ...]
    controller_state_files: tuple[ControllerStateFile, ...] = ()
    cuda_memory_probe: bool = False

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ProbePolicy":
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbeConfigError("probe config must be UTF-8 JSON") from exc
        if (
            not isinstance(value, dict)
            or not {"version", "ownerId", "profiles"}.issubset(value)
            or set(value) - {
                "version", "ownerId", "profiles", "controllerStateFiles",
                "cudaMemoryProbe",
            }
        ):
            raise ProbeConfigError(
                "probe config must contain version, ownerId, profiles, and optional "
                "controllerStateFiles and cudaMemoryProbe"
            )
        if value["version"] != 1:
            raise ProbeConfigError("probe config version must be 1")
        owner_id = _identifier(value["ownerId"], "ownerId")
        entries = value["profiles"]
        if not isinstance(entries, list) or len(entries) > 128:
            raise ProbeConfigError("profiles must be a list of at most 128 entries")
        profiles: list[InstalledProfile] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProbeConfigError("each profile must be an object")
            kind = entry.get("type")
            if kind == "docker":
                if set(entry) != {"id", "type", "container", "expectedIdentity"}:
                    raise ProbeConfigError("docker profiles require exactly id, type, container, and expectedIdentity")
                container = entry["container"]
                if not isinstance(container, str) or not _CONTAINER_NAME.fullmatch(container):
                    raise ProbeConfigError("docker profile container is invalid")
                unit = None
            elif kind == "systemd-user":
                if set(entry) != {"id", "type", "unit", "expectedIdentity"}:
                    raise ProbeConfigError("systemd-user profiles require exactly id, type, unit, and expectedIdentity")
                unit = entry["unit"]
                if not isinstance(unit, str) or not _SYSTEMD_UNIT.fullmatch(unit):
                    raise ProbeConfigError("systemd-user profile unit is invalid")
                container = None
            else:
                raise ProbeConfigError("profile type must be docker or systemd-user")
            expected_identity = entry["expectedIdentity"]
            if not isinstance(expected_identity, str) or not _SHA256_IDENTITY.fullmatch(expected_identity):
                raise ProbeConfigError("expectedIdentity must be a lowercase sha256 digest")
            profiles.append(InstalledProfile(
                id=_identifier(entry["id"], "profile.id"),
                kind=kind,
                expected_identity=expected_identity,
                container=container,
                unit=unit,
            ))
        if len({item.id for item in profiles}) != len(profiles):
            raise ProbeConfigError("profile ids must be unique")
        runtime_keys = [(item.kind, item.container or item.unit) for item in profiles]
        if len(set(runtime_keys)) != len(runtime_keys):
            raise ProbeConfigError("runtime mappings must be unique")
        raw_state_files = value.get("controllerStateFiles", [])
        if not isinstance(raw_state_files, list) or len(raw_state_files) > 128:
            raise ProbeConfigError("controllerStateFiles must be a list of at most 128 entries")
        state_files: list[ControllerStateFile] = []
        for entry in raw_state_files:
            if not isinstance(entry, dict) or set(entry) != {"id", "stateFile"}:
                raise ProbeConfigError("controller state files require exactly id and stateFile")
            controller_id = _identifier(entry["id"], "controllerStateFile.id")
            state_file = entry["stateFile"]
            if not isinstance(state_file, str) or not Path(state_file).is_absolute():
                raise ProbeConfigError("controller stateFile must be an absolute path")
            state_files.append(ControllerStateFile(controller_id, Path(state_file)))
        if len({item.id for item in state_files}) != len(state_files):
            raise ProbeConfigError("controller state file ids must be unique")
        cuda_memory_probe = value.get("cudaMemoryProbe", False)
        if not isinstance(cuda_memory_probe, bool):
            raise ProbeConfigError("cudaMemoryProbe must be boolean")
        return cls(
            owner_id=owner_id,
            profiles=tuple(profiles),
            controller_state_files=tuple(state_files),
            cuda_memory_probe=cuda_memory_probe,
        )


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProbeConfigError(f"{field} is invalid")
    return value


def _secure_read(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    try:
        return read_owner_file(path, label, maximum_bytes=maximum_bytes)
    except SecureFileError as exc:
        raise ProbeConfigError(str(exc)) from exc


def load_policy(path: Path) -> ProbePolicy:
    return ProbePolicy.from_bytes(_secure_read(path, "probe config", maximum_bytes=1024 * 1024))


def load_token(path: Path) -> str:
    raw = _secure_read(path, "probe credential", maximum_bytes=4096)
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise ProbeConfigError("probe credential must be UTF-8") from exc
    if not 32 <= len(value) <= 2048 or "\n" in value or "\r" in value:
        raise ProbeConfigError("probe credential must contain one value of 32-2048 characters")
    return value


def load_controller_state(source: ControllerStateFile) -> dict[str, Any]:
    raw = _secure_read(source.path, f"controller state {source.id}", maximum_bytes=64 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeConfigError(f"controller state {source.id} must be UTF-8 JSON") from exc
    required = {
        "protocolVersion", "controllerId", "mutationId", "leaseId", "fencingToken",
        "brokerEpoch", "controlGeneration", "effectiveMode", "health",
        "appliedAtSafeBoundary",
    }
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or set(value) - (required | {"checkpoint"})
    ):
        raise ProbeConfigError(f"controller state {source.id} has an invalid schema")
    if (
        value["protocolVersion"] != "1.0"
        or value["controllerId"] != source.id
        or not isinstance(value["mutationId"], str)
        or not _MUTATION_ID.fullmatch(value["mutationId"])
        or not isinstance(value["leaseId"], str)
        or not _LEASE_ID.fullmatch(value["leaseId"])
        or not isinstance(value["fencingToken"], str)
        or not _FENCE.fullmatch(value["fencingToken"])
        or not isinstance(value["brokerEpoch"], int)
        or isinstance(value["brokerEpoch"], bool)
        or value["brokerEpoch"] < 1
        or not isinstance(value["controlGeneration"], int)
        or isinstance(value["controlGeneration"], bool)
        or value["controlGeneration"] < 1
        or not isinstance(value["effectiveMode"], str)
        or not _IDENTIFIER.fullmatch(value["effectiveMode"])
        or value["health"] != "healthy"
        or value["appliedAtSafeBoundary"] is not True
    ):
        raise ProbeConfigError(f"controller state {source.id} failed identity validation")
    checkpoint = value.get("checkpoint")
    if checkpoint is not None and (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"runId", "checkpointId", "sha256"}
        or not all(isinstance(checkpoint.get(key), str) and checkpoint[key] for key in ("runId", "checkpointId"))
        or not re.fullmatch(r"[a-f0-9]{64}", checkpoint.get("sha256", ""))
    ):
        raise ProbeConfigError(f"controller state {source.id} checkpoint is invalid")
    return value


class CommandRunner(Protocol):
    def __call__(self, argv: tuple[str, ...]) -> str: ...


class ProcessReader(Protocol):
    def cgroup(self, pid: int) -> str: ...

    def executable_sha256(self, pid: int) -> str: ...


def run_command(argv: tuple[str, ...]) -> str:
    runtime_directory = f"/run/user/{os.geteuid()}"
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LC_ALL": "C",
    }
    if Path(runtime_directory).is_dir():
        environment["XDG_RUNTIME_DIR"] = runtime_directory
        environment["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime_directory}/bus"
        if Path(f"{runtime_directory}/docker.sock").exists():
            environment["DOCKER_HOST"] = f"unix://{runtime_directory}/docker.sock"
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip()[:256] or f"exit {completed.returncode}"
        raise RuntimeError(f"{argv[0]} failed: {message}")
    if len(completed.stdout) > 4 * 1024 * 1024:
        raise RuntimeError(f"{argv[0]} output exceeded 4 MiB")
    return completed.stdout


def run_cuda_memory_probe() -> dict[str, int]:
    """Query a short-lived CUDA process without making this daemon a GPU consumer."""

    helper = Path(__file__).with_name("cuda_memory_probe.py")
    environment = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LC_ALL": "C",
    }
    completed = subprocess.run(
        (sys.executable, str(helper)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip()[:256] or f"exit {completed.returncode}"
        raise RuntimeError(f"CUDA memory probe failed: {message}")
    if len(completed.stdout) > 4096:
        raise RuntimeError("CUDA memory probe output exceeded 4 KiB")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("CUDA memory probe returned invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"allocatableBytes", "addressSpaceTotalBytes"}
        or any(
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
            for field in value
        )
        or value["addressSpaceTotalBytes"] <= 0
        or value["allocatableBytes"] > value["addressSpaceTotalBytes"]
    ):
        raise RuntimeError("CUDA memory probe returned an invalid envelope")
    return value


class LinuxProcessReader:
    def cgroup(self, pid: int) -> str:
        path = Path(f"/proc/{pid}/cgroup")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("cgroup entry is not regular")
            raw = os.read(descriptor, 64 * 1024 + 1)
            if len(raw) > 64 * 1024:
                raise OSError("cgroup entry is too large")
            return raw.decode("utf-8")
        finally:
            os.close(descriptor)

    def executable_sha256(self, pid: int) -> str:
        # Opening /proc/PID/exe binds the digest to the inode actually running,
        # avoiding a pathname swap between inspection and hashing.
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(Path(f"/proc/{pid}/exe"), flags)
        digest = hashlib.sha256()
        try:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        finally:
            os.close(descriptor)
        return f"sha256:{digest.hexdigest()}"


class GenerationStore:
    def __init__(self, path: Path) -> None:
        if not path.is_absolute():
            raise ProbeConfigError("generation path must be absolute")
        self.path = path
        self.lock_path = path.with_name(f"{path.name}.lock")
        self._thread_lock = threading.Lock()

    def next(self) -> int:
        with self._thread_lock:
            lock_descriptor = self._open_lock()
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
                previous = self._read()
                generation = previous + 1
                self._write(generation)
                return generation
            finally:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(lock_descriptor)

    def _open_lock(self) -> int:
        self._validate_parent()
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
            self._validate_descriptor(descriptor, "generation lock")
            return descriptor
        except OSError as exc:
            raise ProbeConfigError("generation lock cannot be opened safely") from exc

    def _validate_parent(self) -> None:
        try:
            metadata = self.path.parent.stat()
        except OSError as exc:
            raise ProbeConfigError("generation directory is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise ProbeConfigError(
                "generation directory must be owned by the service account and not group/world writable"
            )

    @staticmethod
    def _validate_descriptor(descriptor: int, label: str) -> None:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise ProbeConfigError(f"{label} must be regular, service-owned, and mode 0600 or stricter")

    def _read(self) -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            return 0
        except OSError as exc:
            raise ProbeConfigError("generation file cannot be opened safely") from exc
        try:
            self._validate_descriptor(descriptor, "generation file")
            raw = os.read(descriptor, 129)
            if len(raw) > 128:
                raise ProbeConfigError("generation file is invalid")
            value = int(raw.decode("ascii").strip())
            if value < 0:
                raise ValueError
            return value
        except (UnicodeError, ValueError) as exc:
            raise ProbeConfigError("generation file is invalid") from exc
        finally:
            os.close(descriptor)

    def _write(self, value: int) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(temporary, flags, 0o600)
            os.write(descriptor, f"{value}\n".encode("ascii"))
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise ProbeConfigError("generation could not be persisted") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class _Runtime:
    profile: InstalledProfile
    identity_verified: bool
    runtime_identity: str
    cgroup: str | None
    container_id: str | None
    main_pid: int | None
    health: str


class ResourceInventory:
    def __init__(
        self,
        policy: ProbePolicy,
        generation_store: GenerationStore,
        *,
        command: CommandRunner = run_command,
        processes: ProcessReader | None = None,
        meminfo_path: Path = Path("/proc/meminfo"),
        pressure_path: Path = Path("/proc/pressure/memory"),
        monotonic: Callable[[], float] = time.monotonic,
        cuda_memory_probe: Callable[[], dict[str, int]] = run_cuda_memory_probe,
    ) -> None:
        self.policy = policy
        self.generation_store = generation_store
        self.command = command
        self.processes = processes or LinuxProcessReader()
        self.meminfo_path = meminfo_path
        self.pressure_path = pressure_path
        self.monotonic = monotonic
        self.cuda_memory_probe = cuda_memory_probe

    def snapshot(self) -> dict[str, Any]:
        generation = self.generation_store.next()
        metrics, metric_errors = self._host_metrics()
        errors: list[str] = list(metric_errors)
        if self.policy.cuda_memory_probe:
            try:
                cuda_memory = self.cuda_memory_probe()
                allocatable = cuda_memory["allocatableBytes"]
                total = cuda_memory["addressSpaceTotalBytes"]
                if (
                    not isinstance(allocatable, int)
                    or isinstance(allocatable, bool)
                    or not isinstance(total, int)
                    or isinstance(total, bool)
                    or allocatable < 0
                    or total <= 0
                    or allocatable > total
                ):
                    raise ValueError("invalid CUDA memory envelope")
            except Exception:
                metrics["cudaAllocatableBytes"] = None
                metrics["cudaAddressSpaceTotalBytes"] = None
                errors.append("cuda_memory_unobservable")
            else:
                metrics["cudaAllocatableBytes"] = allocatable
                metrics["cudaAddressSpaceTotalBytes"] = total
        runtimes: list[_Runtime] = []
        for profile in self.policy.profiles:
            try:
                runtime = self._docker_runtime(profile) if profile.kind == "docker" else self._systemd_runtime(profile)
            except Exception:
                errors.append(f"runtime_unobservable:{profile.id}")
                continue
            if runtime is not None:
                runtimes.append(runtime)
                if not runtime.identity_verified:
                    errors.append(f"identity_mismatch:{profile.id}")
                elif runtime.health != "healthy":
                    errors.append(f"runtime_unhealthy:{profile.id}")

        controller_states: dict[str, dict[str, Any]] = {}
        for source in self.policy.controller_state_files:
            try:
                controller_states[source.id] = load_controller_state(source)
            except Exception:
                errors.append(f"controller_state_unobservable:{source.id}")

        gpu_rows: list[tuple[int, int | None]] = []
        reported_gpu_memory: int | None = None
        enumeration_failed = False
        try:
            gpu_rows = self._gpu_processes()
            reported_gpu_memory = self._reported_gpu_memory_used()
        except Exception:
            enumeration_failed = True
            errors.append("gpu_inventory_unobservable")

        active: dict[str, dict[str, Any]] = {}
        unknown_pids: set[int] = set()
        memory_by_profile: dict[str, int | None] = {}
        for pid, used_memory in gpu_rows:
            try:
                cgroup = self.processes.cgroup(pid)
            except Exception:
                unknown_pids.add(pid)
                errors.append(f"consumer_unobservable:{pid}")
                continue
            matches = [runtime for runtime in runtimes if self._matches(runtime, pid, cgroup)]
            if len(matches) != 1 or not matches[0].identity_verified:
                unknown_pids.add(pid)
                errors.append(f"consumer_unknown:{pid}")
                continue
            runtime = matches[0]
            if runtime.profile.kind == "systemd-user":
                try:
                    consumer_identity = self.processes.executable_sha256(pid)
                except Exception:
                    unknown_pids.add(pid)
                    errors.append(f"consumer_identity_unobservable:{pid}")
                    continue
                if not hmac.compare_digest(consumer_identity, runtime.profile.expected_identity):
                    unknown_pids.add(pid)
                    errors.append(f"consumer_identity_mismatch:{pid}")
                    continue
            current = memory_by_profile.get(runtime.profile.id, 0)
            memory_by_profile[runtime.profile.id] = None if current is None or used_memory is None else current + used_memory
            active[runtime.profile.id] = {
                "health": runtime.health,
                "identityVerified": True,
                "runtimeIdentity": runtime.runtime_identity,
                "ownerId": self.policy.owner_id,
            }

        for profile_id, value in memory_by_profile.items():
            active[profile_id]["gpuMemoryBytes"] = value
        active_profiles = sorted(active)
        unknown = len(unknown_pids) + (1 if enumeration_failed else 0)
        attributed_gpu_memory = (
            sum(value for _pid, value in gpu_rows if value is not None)
            if all(value is not None for _pid, value in gpu_rows)
            else None
        )
        residual_gpu_memory: int | None = None
        reconciled = False
        if reported_gpu_memory is not None:
            if attributed_gpu_memory is None:
                unknown += 1
                errors.append("gpu_memory_attribution_incomplete")
            else:
                residual_gpu_memory = reported_gpu_memory - attributed_gpu_memory
                if abs(residual_gpu_memory) > _GPU_MEMORY_RECONCILIATION_TOLERANCE_BYTES:
                    unknown += 1
                    errors.append("gpu_memory_residual_unattributed")
                else:
                    reconciled = True
        healthy = not errors and all(value["health"] == "healthy" for value in active.values())
        return {
            "health": "healthy" if healthy else "degraded",
            "generation": generation,
            "unknownConsumers": unknown,
            "activeProfiles": active_profiles,
            "profiles": {key: active[key] for key in active_profiles},
            "controllerStates": controller_states,
            "gpuMemory": {
                "reportedUsedBytes": reported_gpu_memory,
                "attributedBytes": attributed_gpu_memory,
                "residualBytes": residual_gpu_memory,
                "reconciled": reconciled,
                "toleranceBytes": _GPU_MEMORY_RECONCILIATION_TOLERANCE_BYTES,
            },
            "metrics": metrics,
            "observabilityErrors": sorted(set(errors)),
        }

    def _docker_runtime(self, profile: InstalledProfile) -> _Runtime | None:
        assert profile.container is not None
        raw = self.command(("docker", "inspect", profile.container))
        value = json.loads(raw)
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
            raise RuntimeError("unexpected docker inspect result")
        item = value[0]
        state = item.get("State")
        container_id = item.get("Id")
        image_id = item.get("Image")
        if not isinstance(state, dict) or state.get("Running") is not True:
            return None
        if (
            not isinstance(container_id, str)
            or not re.fullmatch(r"[a-f0-9]{64}", container_id)
            or not isinstance(image_id, str)
            or not _SHA256_IDENTITY.fullmatch(image_id)
        ):
            raise RuntimeError("docker identity is invalid")
        verified = hmac.compare_digest(image_id, profile.expected_identity)
        return _Runtime(
            profile=profile,
            identity_verified=verified,
            runtime_identity=f"oci-image:{image_id};container:{container_id}",
            cgroup=None,
            container_id=container_id,
            main_pid=None,
            health="healthy" if verified else "unhealthy",
        )

    def _systemd_runtime(self, profile: InstalledProfile) -> _Runtime | None:
        assert profile.unit is not None
        raw = self.command((
            "systemctl", "--user", "show", profile.unit,
            "--property=ActiveState", "--property=SubState", "--property=MainPID", "--property=ControlGroup",
        ))
        fields: dict[str, str] = {}
        for line in raw.splitlines():
            if "=" not in line:
                raise RuntimeError("invalid systemctl output")
            key, value = line.split("=", 1)
            fields[key] = value
        if set(fields) != {"ActiveState", "SubState", "MainPID", "ControlGroup"}:
            raise RuntimeError("incomplete systemctl output")
        if fields["ActiveState"] != "active":
            return None
        pid = int(fields["MainPID"])
        cgroup = fields["ControlGroup"]
        if pid <= 0 or not cgroup.startswith("/") or ".." in cgroup.split("/"):
            raise RuntimeError("invalid systemd runtime identity")
        actual_identity = self.processes.executable_sha256(pid)
        if not _SHA256_IDENTITY.fullmatch(actual_identity):
            raise RuntimeError("invalid executable identity")
        verified = hmac.compare_digest(actual_identity, profile.expected_identity)
        return _Runtime(
            profile=profile,
            identity_verified=verified,
            runtime_identity=f"systemd-exe:{actual_identity};unit:{profile.unit}",
            cgroup=cgroup.rstrip("/") or "/",
            container_id=None,
            main_pid=pid,
            health="healthy" if verified and fields["SubState"] == "running" else "unhealthy",
        )

    def _gpu_processes(self) -> list[tuple[int, int | None]]:
        compute_raw = self.command((
            "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
        ))
        values: dict[int, int | None] = {}
        for line in compute_raw.splitlines():
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 2:
                raise RuntimeError("invalid nvidia-smi output")
            pid = int(fields[0])
            if pid <= 0:
                raise RuntimeError("invalid GPU process id")
            memory = None if fields[1] in {"N/A", "[N/A]"} else int(fields[1]) * 1024 * 1024
            if memory is not None and memory < 0:
                raise RuntimeError("invalid GPU memory value")
            values[pid] = memory

        # The selective query above covers compute contexts. NVIDIA pmon is a
        # second, independent inventory that also includes graphics contexts.
        # Union the views so an unexpected graphics process cannot disappear
        # behind an otherwise empty compute query.
        pmon_raw = self.command(("nvidia-smi", "pmon", "-c", "1", "-s", "m"))
        for line in pmon_raw.splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) < 2:
                raise RuntimeError("invalid nvidia-smi pmon output")
            if fields[1] == "-":
                continue
            pid = int(fields[1])
            if pid <= 0:
                raise RuntimeError("invalid GPU process id")
            memory = None if len(fields) < 4 or fields[3] in {"-", "N/A", "[N/A]"} else int(fields[3]) * 1024 * 1024
            if memory is not None and memory < 0:
                raise RuntimeError("invalid pmon GPU memory value")
            if pid not in values or values[pid] is None:
                values[pid] = memory
        return sorted(values.items())

    def _reported_gpu_memory_used(self) -> int | None:
        try:
            raw = self.command((
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            ))
        except Exception:
            return None
        total = 0
        found = False
        for line in raw.splitlines():
            value = line.strip()
            if not value or value in {"N/A", "[N/A]"}:
                continue
            memory = int(value)
            if memory < 0:
                raise RuntimeError("invalid reported GPU memory value")
            total += memory * 1024 * 1024
            found = True
        return total if found else None

    @staticmethod
    def _matches(runtime: _Runtime, pid: int, cgroup: str) -> bool:
        if runtime.container_id is not None:
            candidates = {match.group(1) for match in _CONTAINER_ID.finditer(cgroup)}
            # Linux cgroups commonly expose either the full OCI container ID
            # or Docker's canonical 12-character short ID. Accept no other
            # prefix length: a partial 13-63 character string is neither an
            # immutable identity nor a documented short identifier.
            return any(
                candidate == runtime.container_id
                or (len(candidate) == 12 and runtime.container_id.startswith(candidate))
                for candidate in candidates
            )
        if runtime.cgroup is not None:
            base = runtime.cgroup.rstrip("/")
            paths = []
            for line in cgroup.splitlines():
                fields = line.split(":", 2)
                if len(fields) == 3 and fields[2].startswith("/"):
                    paths.append(fields[2].rstrip("/") or "/")
            return any(path == base or path.startswith(f"{base}/") for path in paths)
        return False

    def _host_metrics(self) -> tuple[dict[str, Any], list[str]]:
        values: dict[str, int] = {}
        errors: list[str] = []
        try:
            raw = self.meminfo_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                name, value = line.split(":", 1)
                fields = value.strip().split()
                if len(fields) != 2 or fields[1] != "kB":
                    continue
                values[name] = int(fields[0]) * 1024
        except (OSError, UnicodeError, ValueError):
            errors.append("uma_memory_unobservable")
        pressure: float | None = None
        try:
            raw = self.pressure_path.read_text(encoding="utf-8")
            line = next(item for item in raw.splitlines() if item.startswith("some "))
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            pressure = float(fields["avg10"])
            if pressure < 0:
                raise ValueError
        except (OSError, UnicodeError, ValueError, KeyError, StopIteration):
            errors.append("memory_psi_unobservable")
        metrics = {
            "umaTotalBytes": values.get("MemTotal"),
            "umaAvailableBytes": values.get("MemAvailable"),
            "swapFreeBytes": values.get("SwapFree"),
            "memoryPressureSomeAvg10": pressure,
            "sampledAtMonotonic": float(self.monotonic()),
        }
        if metrics["umaTotalBytes"] is None or metrics["umaAvailableBytes"] is None:
            if "uma_memory_unobservable" not in errors:
                errors.append("uma_memory_unobservable")
        return metrics, errors


@dataclass(frozen=True)
class ProbeServerConfig:
    bind: str
    port: int
    token: str

    def __post_init__(self) -> None:
        try:
            address = ipaddress.ip_address(self.bind)
        except ValueError as exc:
            raise ProbeConfigError("probe bind must be a numeric loopback address") from exc
        if not address.is_loopback:
            raise ProbeConfigError("probe bind must be loopback")
        if not 1 <= self.port <= 65535:
            raise ProbeConfigError("probe port must be 1-65535")
        if not 32 <= len(self.token) <= 2048 or "\n" in self.token or "\r" in self.token:
            raise ProbeConfigError("probe token must contain 32-2048 characters")


class ProbeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: ProbeServerConfig, inventory: ResourceInventory) -> None:
        self.config = config
        self.inventory = inventory
        if ":" in config.bind:
            self.address_family = socket.AF_INET6
        super().__init__((config.bind, config.port), ProbeHandler)


class ProbeHandler(BaseHTTPRequestHandler):
    server: ProbeHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"spark-resource-probe {self.client_address[0]} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        try:
            peer = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            self._json(HTTPStatus.FORBIDDEN, {"error": {"code": "loopback_required"}})
            return
        if not peer.is_loopback:
            self._json(HTTPStatus.FORBIDDEN, {"error": {"code": "loopback_required"}})
            return
        expected = f"Bearer {self.server.config.token}"
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self._json(HTTPStatus.UNAUTHORIZED, {"error": {"code": "unauthorized"}})
            return
        if self.path != "/v1/resource-snapshot":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found"}})
            return
        if self.headers.get("Content-Length") not in {None, "0"}:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "body_not_allowed"}})
            return
        try:
            value = self.server.inventory.snapshot()
        except Exception:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": {"code": "snapshot_unavailable"}})
            return
        self._json(HTTPStatus.OK, value)

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    config_path = Path(os.environ.get("SPARK_RESOURCE_PROBE_CONFIG", ""))
    token_path = Path(os.environ.get("SPARK_RESOURCE_PROBE_TOKEN_FILE", ""))
    generation_path = Path(os.environ.get("SPARK_RESOURCE_PROBE_GENERATION_FILE", ""))
    if not all(str(path) for path in (config_path, token_path, generation_path)):
        raise SystemExit(
            "SPARK_RESOURCE_PROBE_CONFIG, SPARK_RESOURCE_PROBE_TOKEN_FILE, and "
            "SPARK_RESOURCE_PROBE_GENERATION_FILE are required"
        )
    try:
        policy = load_policy(config_path)
        token = load_token(token_path)
        generation = GenerationStore(generation_path)
        # Validate persistence before opening a network listener. This consumes
        # one generation but guarantees every served value is durable.
        generation.next()
        config = ProbeServerConfig(
            bind=os.environ.get("SPARK_RESOURCE_PROBE_BIND", "127.0.0.1"),
            port=int(os.environ.get("SPARK_RESOURCE_PROBE_PORT", "8791")),
            token=token,
        )
    except (ProbeConfigError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    server = ProbeHTTPServer(config, ResourceInventory(policy, generation))
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
