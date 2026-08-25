from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .store import Store


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
_SERVICE_CLASSES = {"interactive", "batch", "background", "training", "system"}
_LEASE_MODES = {"exclusive", "shared-certified", "permit", "none"}


class AdmissionDeferred(RuntimeError):
    def __init__(self, code: str, message: str, *, retry_after_seconds: float = 2.0, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_seconds = retry_after_seconds
        self.detail = detail or {}


@dataclass(frozen=True)
class ExecutionPlan:
    profile_id: str
    route_id: str | None
    resource_group: str | None
    service_class: str
    lease_mode: str
    estimated_memory_gb: int
    preemption_mode: str = "none"
    route_reason: str = "single_profile"
    verify_profile_active: bool = False

    def __post_init__(self) -> None:
        if self.service_class not in _SERVICE_CLASSES:
            raise ValueError("invalid service class")
        if self.lease_mode not in _LEASE_MODES:
            raise ValueError("invalid lease mode")
        if self.resource_group is None and self.lease_mode != "none":
            raise ValueError("resource-free plans must use lease mode none")
        if not 0 <= self.estimated_memory_gb <= 1024:
            raise ValueError("estimated memory must be 0-1024 GiB")

    def public(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "routeId": self.route_id,
            "resourceGroup": self.resource_group,
            "serviceClass": self.service_class,
            "leaseMode": self.lease_mode,
            "estimatedMemoryGb": self.estimated_memory_gb,
            "preemptionMode": self.preemption_mode,
            "routeReason": self.route_reason,
            "verifyProfileActive": self.verify_profile_active,
        }


class ExecutionControl:
    """Cancellation plus cooperative checkpoint/yield signalling.

    It remains callable so protocol-1.0 executors that only understand a
    cancellation callback keep working. Managed training adapters may inspect
    ``yield_requested`` and publish a verified checkpoint before returning.
    """

    def __init__(self, cancelled: Callable[[], bool]) -> None:
        self._cancelled = cancelled
        self._yield_requested = threading.Event()
        self._yield_lock = threading.Lock()
        self._yield_reason: str | None = None

    def __call__(self) -> bool:
        return self._cancelled()

    def request_yield(self, reason: str) -> None:
        with self._yield_lock:
            if self._yield_reason is None:
                self._yield_reason = reason
                self._yield_requested.set()

    def yield_requested(self) -> bool:
        return self._yield_requested.is_set()

    @property
    def yield_reason(self) -> str | None:
        with self._yield_lock:
            return self._yield_reason


@dataclass(frozen=True)
class ControllerPolicy:
    id: str
    profile_id: str
    endpoint: str
    token_file: Path
    throttle_for: tuple[str, ...]
    normal_mode: str
    throttled_mode: str
    priority: int
    workload_kind: str = "background"
    requires_checkpoint: bool = False


@dataclass(frozen=True)
class ResourcePolicy:
    host_reserve_gb: float = 16.0
    maximum_memory_pressure_avg10: float = 5.0
    enforce_memory_admission: bool = False
    require_probe: bool = False
    probe_endpoint: str | None = None
    probe_token_file: Path | None = None
    controllers: tuple[ControllerPolicy, ...] = ()
    shared_certifications: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_file(cls, path: Path) -> "ResourcePolicy":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("resource policy must be a version 1 object")
        allowed = {
            "version", "hostReserveGb", "maximumMemoryPressureAvg10",
            "enforceMemoryAdmission", "probe", "controllers", "sharedCertifications",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"resource policy has unknown fields: {sorted(unknown)}")
        reserve = raw.get("hostReserveGb", 16)
        pressure = raw.get("maximumMemoryPressureAvg10", 5)
        enforce = raw.get("enforceMemoryAdmission", False)
        if not isinstance(reserve, (int, float)) or isinstance(reserve, bool) or not 1 <= reserve <= 128:
            raise ValueError("hostReserveGb must be 1-128")
        if not isinstance(pressure, (int, float)) or isinstance(pressure, bool) or not 0 <= pressure <= 100:
            raise ValueError("maximumMemoryPressureAvg10 must be 0-100")
        if not isinstance(enforce, bool):
            raise ValueError("enforceMemoryAdmission must be boolean")

        probe = raw.get("probe")
        probe_endpoint: str | None = None
        probe_token_file: Path | None = None
        require_probe = False
        if probe is not None:
            if not isinstance(probe, dict) or set(probe) - {"endpoint", "tokenFile", "required"}:
                raise ValueError("probe must contain only endpoint, tokenFile, and required")
            probe_endpoint = _loopback_url(probe.get("endpoint"), "probe.endpoint")
            probe_token_file = _credential_path(probe.get("tokenFile"), "probe.tokenFile")
            require_probe = probe.get("required", True)
            if not isinstance(require_probe, bool):
                raise ValueError("probe.required must be boolean")

        controllers: list[ControllerPolicy] = []
        raw_controllers = raw.get("controllers", [])
        if not isinstance(raw_controllers, list) or len(raw_controllers) > 32:
            raise ValueError("controllers must be a list of at most 32 entries")
        for item in raw_controllers:
            if not isinstance(item, dict) or set(item) - {
                "id", "profileId", "endpoint", "tokenFile", "throttleFor",
                "normalMode", "throttledMode", "priority",
                "requiresCheckpoint",
                "workloadKind",
            }:
                raise ValueError("controller contains unknown fields")
            controller_id = _identifier(item.get("id"), "controller.id")
            profile_id = _identifier(item.get("profileId"), "controller.profileId")
            throttle_for = item.get("throttleFor", ["interactive"])
            if not isinstance(throttle_for, list) or not throttle_for or any(value not in _SERVICE_CLASSES for value in throttle_for):
                raise ValueError("controller.throttleFor contains an invalid service class")
            priority = item.get("priority", 20)
            if not isinstance(priority, int) or isinstance(priority, bool) or not 0 <= priority <= 100:
                raise ValueError("controller.priority must be 0-100")
            workload_kind = item.get("workloadKind")
            if workload_kind not in {"training", "inference", "background"}:
                raise ValueError("controller.workloadKind must be training, inference, or background")
            requires_checkpoint = item.get("requiresCheckpoint", workload_kind == "training")
            if not isinstance(requires_checkpoint, bool):
                raise ValueError("controller.requiresCheckpoint must be boolean")
            if workload_kind == "training" and not requires_checkpoint:
                raise ValueError("training controllers must require checkpoint proof")
            controllers.append(ControllerPolicy(
                id=controller_id,
                profile_id=profile_id,
                endpoint=_loopback_url(item.get("endpoint"), "controller.endpoint"),
                token_file=_credential_path(item.get("tokenFile"), "controller.tokenFile"),
                throttle_for=tuple(dict.fromkeys(throttle_for)),
                normal_mode=_identifier(item.get("normalMode", "normal"), "controller.normalMode"),
                throttled_mode=_identifier(item.get("throttledMode", "interactive-boost"), "controller.throttledMode"),
                priority=priority,
                requires_checkpoint=requires_checkpoint,
                workload_kind=workload_kind,
            ))
        if len({item.id for item in controllers}) != len(controllers):
            raise ValueError("controller ids must be unique")

        certifications: list[tuple[str, str]] = []
        raw_certifications = raw.get("sharedCertifications", [])
        if not isinstance(raw_certifications, list) or len(raw_certifications) > 128:
            raise ValueError("sharedCertifications must be a list")
        if raw_certifications:
            raise ValueError(
                "shared certifications are not accepted yet; this release requires verified exclusive release"
            )
        for item in raw_certifications:
            if not isinstance(item, dict) or set(item) != {"profiles"}:
                raise ValueError("shared certification must contain exactly profiles")
            profiles = item["profiles"]
            if not isinstance(profiles, list) or len(profiles) != 2:
                raise ValueError("shared certification profiles must contain exactly two ids")
            pair = tuple(sorted(_identifier(value, "sharedCertification.profile") for value in profiles))
            certifications.append((pair[0], pair[1]))
        return cls(
            host_reserve_gb=float(reserve),
            maximum_memory_pressure_avg10=float(pressure),
            enforce_memory_admission=enforce,
            require_probe=require_probe,
            probe_endpoint=probe_endpoint,
            probe_token_file=probe_token_file,
            controllers=tuple(controllers),
            shared_certifications=tuple(dict.fromkeys(certifications)),
        )


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} is invalid")
    return value


def _loopback_url(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
        raise ValueError(f"{field_name} must be an HTTP loopback URL without credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain a query or fragment")
    return value.rstrip("/")


def _credential_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{field_name} must be an absolute path")
    return Path(value)


class HostProbe:
    def snapshot(self) -> dict[str, Any]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                name, raw = line.split(":", 1)
                values[name] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            values = {}
        pressure: float | None = None
        try:
            for line in Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines():
                if line.startswith("some "):
                    fields = dict(value.split("=", 1) for value in line.split()[1:])
                    pressure = float(fields["avg10"])
                    break
        except (OSError, ValueError, KeyError):
            pressure = None
        return {
            "availableMemoryBytes": values.get("MemAvailable"),
            "totalMemoryBytes": values.get("MemTotal"),
            "swapFreeBytes": values.get("SwapFree"),
            "memoryPressureAvg10": pressure,
            "sampledAtMonotonic": time.monotonic(),
        }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class ResourceControlClient:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    @staticmethod
    def _token(path: Path) -> str:
        try:
            stat = path.stat()
            if stat.st_uid != os.geteuid() or stat.st_mode & 0o077:
                raise AdmissionDeferred(
                    "controller_credential_invalid",
                    "resource controller credential must be owned by the service account and mode 0600 or stricter",
                )
            value = path.read_text(encoding="utf-8").strip()
        except AdmissionDeferred:
            raise
        except OSError as exc:
            raise AdmissionDeferred(
                "controller_credential_invalid", "resource controller credential cannot be read"
            ) from exc
        if len(value) < 32:
            raise AdmissionDeferred("controller_credential_invalid", "resource controller credential is missing or too short")
        return value

    def snapshot(self, endpoint: str, token_file: Path) -> dict[str, Any]:
        return self._request("GET", f"{endpoint}/v1/resource-snapshot", token_file, None)

    def set_mode(
        self,
        controller: ControllerPolicy,
        *,
        mode: str,
        lease_id: str,
        fencing_token: str,
        broker_epoch: int,
        generation: int,
        reason: str,
    ) -> dict[str, Any]:
        payload = {
            "protocolVersion": "1.0",
            "leaseId": lease_id,
            "fencingToken": fencing_token,
            "brokerEpoch": broker_epoch,
            "controlGeneration": generation,
            "targetMode": mode,
            "reason": reason,
        }
        value = self._request("POST", f"{controller.endpoint}/v1/resource-mode", controller.token_file, payload)
        if (
            value.get("leaseId") != lease_id
            or value.get("fencingToken") != fencing_token
            or value.get("brokerEpoch") != broker_epoch
        ):
            raise AdmissionDeferred("controller_fence_mismatch", f"controller {controller.id} returned a mismatched lease fence")
        if value.get("acknowledgedGeneration") != generation or value.get("effectiveMode") != mode:
            raise AdmissionDeferred("controller_ack_mismatch", f"controller {controller.id} did not acknowledge the requested generation")
        if value.get("health") != "healthy" or value.get("appliedAtSafeBoundary") is not True:
            raise AdmissionDeferred("controller_unhealthy", f"controller {controller.id} is not healthy")
        snapshot_generation = value.get("snapshotGeneration")
        if not isinstance(snapshot_generation, int) or isinstance(snapshot_generation, bool) or snapshot_generation < 0:
            raise AdmissionDeferred("controller_ack_mismatch", f"controller {controller.id} omitted snapshot generation")
        if controller.requires_checkpoint and mode == controller.throttled_mode:
            checkpoint = value.get("checkpoint")
            if (
                not isinstance(checkpoint, dict)
                or set(checkpoint) != {"runId", "checkpointId", "sha256"}
                or not all(isinstance(checkpoint.get(key), str) and checkpoint[key] for key in ("runId", "checkpointId"))
                or not re.fullmatch(r"[a-f0-9]{64}", checkpoint.get("sha256", ""))
            ):
                raise AdmissionDeferred(
                    "controller_checkpoint_missing",
                    f"controller {controller.id} did not prove a durable checkpoint boundary",
                )
        return value

    def takeover_mode(
        self,
        controller: ControllerPolicy,
        *,
        previous_lease_id: str,
        previous_fencing_token: str,
        recovery_fencing_token: str,
        broker_epoch: int,
        generation: int,
        mode: str,
    ) -> dict[str, Any]:
        payload = {
            "protocolVersion": "1.0",
            "previousLeaseId": previous_lease_id,
            "previousFencingToken": previous_fencing_token,
            "recoveryFencingToken": recovery_fencing_token,
            "brokerEpoch": broker_epoch,
            "controlGeneration": generation,
            "targetMode": mode,
            "reason": f"restart-recovery:{previous_lease_id}",
        }
        value = self._request("POST", f"{controller.endpoint}/v1/resource-takeover", controller.token_file, payload)
        if (
            value.get("previousLeaseId") != previous_lease_id
            or value.get("previousFencingToken") != previous_fencing_token
            or value.get("recoveryFencingToken") != recovery_fencing_token
            or value.get("brokerEpoch") != broker_epoch
            or value.get("acknowledgedGeneration") != generation
            or value.get("effectiveMode") != mode
            or value.get("health") != "healthy"
            or value.get("appliedAtSafeBoundary") is not True
            or not isinstance(value.get("snapshotGeneration"), int)
        ):
            raise AdmissionDeferred("controller_takeover_mismatch", f"controller {controller.id} rejected recovery takeover")
        return value

    def _request(self, method: str, url: str, token_file: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={"Authorization": f"Bearer {self._token(token_file)}", "Content-Type": "application/json"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise AdmissionDeferred("controller_http_error", f"resource controller returned HTTP {response.status}")
                raw = response.read(1024 * 1024 + 1)
        except AdmissionDeferred:
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise AdmissionDeferred("controller_unavailable", "resource controller is unavailable") from exc
        if len(raw) > 1024 * 1024:
            raise AdmissionDeferred("controller_response_too_large", "resource controller response exceeds 1 MiB")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdmissionDeferred("controller_invalid_response", "resource controller returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise AdmissionDeferred("controller_invalid_response", "resource controller response must be an object")
        return value


@dataclass
class LeaseHandle:
    lease: dict[str, Any]
    throttled: list[dict[str, Any]] = field(default_factory=list)


class ResourceCoordinator:
    def __init__(
        self,
        *,
        store: Store,
        data_root: Path,
        policy: ResourcePolicy | None = None,
        host_probe: HostProbe | None = None,
        control_client: ResourceControlClient | None = None,
        lock_path: Path | None = None,
        epoch_path: Path | None = None,
    ) -> None:
        self.store = store
        self.data_root = data_root
        self.policy = policy or ResourcePolicy()
        self.host_probe = host_probe or HostProbe()
        self.control_client = control_client or ResourceControlClient()
        self.lock_path = lock_path or (data_root / "coordinator.lock")
        self.epoch_path = epoch_path or Path(f"{self.lock_path}.epoch")
        self.instance_id = f"broker_{uuid.uuid4().hex}"
        self.epoch = 0
        self._lock_handle: Any = None
        self._quarantine_reason: str | None = None

    def start(self) -> None:
        lock_path = self.lock_path
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_mode & 0o077
            ):
                os.close(descriptor)
                raise RuntimeError(
                    "coordinator lock must be a regular file owned by the service account and mode 0600"
                )
            self._lock_handle = os.fdopen(descriptor, "a+b")
        except OSError as exc:
            raise RuntimeError("coordinator lock cannot be opened safely") from exc
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError("another broker instance owns this host resource coordinator") from exc
        try:
            self._reconcile_after_lock()
        except BaseException:
            self.stop()
            raise

    def _reconcile_after_lock(self) -> None:
        self.epoch = self._advance_host_epoch()
        stale = self.store.mark_active_leases_unknown()
        for lease in stale:
            try:
                snapshot = self._combined_snapshot()
            except AdmissionDeferred as exc:
                self._quarantine_reason = f"stale lease {lease['id']} could not reconcile: {exc.code}"
                continue
            requires_absence = lease["mode"] == "exclusive" or bool(lease.get("throttle"))
            if not self._released_profile_is_verified(snapshot, lease["profileId"], requires_absence=requires_absence):
                self._quarantine_reason = f"stale lease {lease['id']} has no verified resource release"
                continue
            if lease.get("throttle"):
                try:
                    expected_generation = self._recover_recorded_throttles(lease)
                    restored = self._combined_snapshot()
                except AdmissionDeferred as exc:
                    self._quarantine_reason = f"stale lease {lease['id']} recovery takeover failed: {exc.code}"
                    continue
                if not isinstance(restored.get("probeGeneration"), int) or restored["probeGeneration"] < expected_generation:
                    self._quarantine_reason = f"stale lease {lease['id']} recovery snapshot is stale"
                    continue
                if not self._released_profile_is_verified(restored, lease["profileId"], requires_absence=True):
                    self._quarantine_reason = f"stale lease {lease['id']} restore could not be verified"
                    continue
            self.store.set_resource_lease_status(lease["id"], "released")
        self.store.interrupt_uncertain_jobs()

    def _advance_host_epoch(self) -> int:
        epoch_path = self.epoch_path
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(epoch_path, flags)
        except FileNotFoundError:
            raw = "0"
        except OSError as exc:
            raise RuntimeError("host coordinator epoch file cannot be opened safely") from exc
        else:
            try:
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_mode & 0o077
                ):
                    raise RuntimeError(
                        "host coordinator epoch must be a regular file owned by the service account and mode 0600"
                    )
                with os.fdopen(descriptor, "r", encoding="ascii") as stream:
                    descriptor = -1
                    raw = stream.read(128).strip()
                    if stream.read(1):
                        raise ValueError
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        try:
            previous = int(raw)
            if previous < 0:
                raise ValueError
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError("host coordinator epoch file is invalid") from exc
        epoch = self.store.begin_broker_epoch(self.instance_id, minimum_epoch=previous + 1)
        temporary = epoch_path.with_name(f".{epoch_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(f"{epoch}\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, epoch_path)
            directory = os.open(epoch_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("host coordinator epoch could not be persisted") from exc
        return epoch

    @property
    def quarantined(self) -> bool:
        return self._quarantine_reason is not None

    def stop(self) -> None:
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    def status(self) -> dict[str, Any]:
        active = self.store.resource_leases(active_only=True)
        resource_state = "quarantined" if self._quarantine_reason else "ready"
        snapshot: dict[str, Any] | None = None
        if resource_state == "ready" and self.policy.require_probe:
            try:
                snapshot = self._combined_snapshot()
            except AdmissionDeferred:
                resource_state = "unavailable"
            else:
                if snapshot.get("probeHealthy") is not True or snapshot.get("unknownConsumers") != 0:
                    resource_state = "unavailable"
        return {
            "brokerEpoch": self.epoch,
            "resourceState": resource_state,
            "quarantineReason": self._quarantine_reason,
            "activeLeases": [self._public_lease(item) for item in active],
            "host": snapshot or self.host_probe.snapshot(),
        }

    def routing_context(self) -> dict[str, Any]:
        """Return observed, non-authoritative hints for profile selection.

        Admission re-samples and remains the safety boundary. A stale routing
        hint may cost a model switch, but can never grant a lease.
        """
        try:
            return self._combined_snapshot()
        except AdmissionDeferred:
            return {"probeHealthy": False, "activeProfiles": [], "profiles": {}}

    def acquire(self, job: dict[str, Any], plan: ExecutionPlan, cancelled: Callable[[], bool]) -> LeaseHandle | None:
        if plan.resource_group is None or plan.lease_mode == "none":
            return None
        if cancelled():
            raise AdmissionDeferred("cancelled", "job was cancelled before admission", retry_after_seconds=0)
        if self._quarantine_reason:
            raise AdmissionDeferred("resource_quarantined", self._quarantine_reason, retry_after_seconds=10)
        if not self.policy.require_probe or not self.policy.probe_endpoint or not self.policy.probe_token_file:
            raise AdmissionDeferred(
                "resource_probe_required",
                "GPU admission requires an administrator-configured healthy resource probe",
                retry_after_seconds=10,
            )
        if self.store.resource_leases(active_only=True):
            raise AdmissionDeferred("resource_busy", "resource group has an active or unreconciled lease")

        snapshot = self._combined_snapshot()
        self._validate_snapshot(snapshot, plan, check_capacity=False)
        decision = {"before": snapshot, "plan": plan.public(), "reason": plan.route_reason}
        lease = self.store.create_resource_lease(
            job_id=job["id"], broker_epoch=self.epoch, resource_group=plan.resource_group,
            profile_id=plan.profile_id, route_id=plan.route_id, service_class=plan.service_class,
            mode=plan.lease_mode, estimated_memory_gb=plan.estimated_memory_gb,
            throttle=[], decision=decision,
        )
        handle = LeaseHandle(lease=lease)
        try:
            for generation, controller in enumerate(self._controllers_for(plan), start=1):
                throttle_record = {
                    "controllerId": controller.id,
                    "normalMode": controller.normal_mode,
                    "generation": generation,
                    "state": "requested",
                }
                # Journal the compensating action before asking an external
                # controller to mutate training or inference state. A crash
                # after the controller applies the request can then be
                # recovered idempotently from this record.
                handle.throttled.append(throttle_record)
                self.store.set_resource_lease_status(
                    lease["id"], "acquiring", throttle=handle.throttled
                )
                acknowledgement = self.control_client.set_mode(
                    controller,
                    mode=controller.throttled_mode,
                    lease_id=lease["id"],
                    fencing_token=lease["fencingToken"],
                    broker_epoch=lease["brokerEpoch"],
                    generation=generation,
                    reason=f"admit:{plan.service_class}:{job['id']}",
                )
                throttle_record["state"] = "applied"
                throttle_record["acknowledgement"] = acknowledgement
                self.store.set_resource_lease_status(lease["id"], "acquiring", throttle=handle.throttled)
            after = self._combined_snapshot()
            expected_generation = max(
                (int(item["acknowledgement"]["snapshotGeneration"]) for item in handle.throttled),
                default=None,
            )
            if expected_generation is not None:
                observed_generation = after.get("probeGeneration")
                if not isinstance(observed_generation, int) or observed_generation < expected_generation:
                    raise AdmissionDeferred(
                        "resource_snapshot_stale",
                        "resource snapshot does not include the acknowledged controller generation",
                    )
            self._validate_snapshot(after, plan, check_capacity=True)
            self._validate_compatibility(after, plan)
            decision["afterThrottle"] = after
            self.store.set_resource_lease_status(lease["id"], "active", throttle=handle.throttled, decision=decision)
            handle.lease = self.store.get_resource_lease(lease["id"]) or lease
            return handle
        except BaseException:
            try:
                self._restore_handle(handle)
            except Exception as restore_exc:
                self.store.set_resource_lease_status(
                    lease["id"], "unknown", throttle=handle.throttled, decision=decision
                )
                code = restore_exc.code if isinstance(restore_exc, AdmissionDeferred) else "controller_restore_failed"
                self._quarantine_reason = f"lease {lease['id']} admission rollback failed: {code}"
                raise AdmissionDeferred(
                    "admission_rollback_failed",
                    "resource admission failed and the prior throttle state could not be verified",
                    retry_after_seconds=10,
                    detail={"restoreCode": code},
                ) from restore_exc
            self.store.set_resource_lease_status(
                lease["id"], "denied", throttle=handle.throttled, decision=decision
            )
            raise

    def release(self, handle: LeaseHandle | None) -> None:
        if handle is None:
            return
        self.store.set_resource_lease_status(handle.lease["id"], "releasing", throttle=handle.throttled)
        try:
            requires_absence = handle.lease["mode"] == "exclusive" or bool(handle.throttled)
            if requires_absence:
                before_restore = self._combined_snapshot()
                if not self._released_profile_is_verified(
                    before_restore, handle.lease["profileId"], requires_absence=True
                ):
                    raise AdmissionDeferred(
                        "release_unverified",
                        "leased profile is still active; preserving background throttle state",
                    )
            self._restore_handle(handle)
            if requires_absence:
                snapshot = self._combined_snapshot()
                if not self._released_profile_is_verified(
                    snapshot, handle.lease["profileId"], requires_absence=True
                ):
                    raise AdmissionDeferred("release_unverified", "exclusive runtime release was not verified")
        except Exception as exc:
            code = exc.code if isinstance(exc, AdmissionDeferred) else "controller_restore_failed"
            self.store.set_resource_lease_status(handle.lease["id"], "unknown", throttle=handle.throttled)
            self._quarantine_reason = f"lease {handle.lease['id']} release failed: {code}"
            if isinstance(exc, AdmissionDeferred):
                raise
            raise AdmissionDeferred(
                "release_restore_failed", "resource throttle restoration failed during release",
                retry_after_seconds=10,
            ) from exc
        self.store.set_resource_lease_status(handle.lease["id"], "released", throttle=handle.throttled)

    def verify_activation(self, handle: LeaseHandle | None, plan: ExecutionPlan) -> None:
        if handle is None:
            return
        snapshot = self._combined_snapshot()
        self._validate_snapshot(snapshot, plan, check_capacity=True)
        self._validate_compatibility(snapshot, plan)
        if plan.verify_profile_active and plan.profile_id not in snapshot.get("activeProfiles", []):
            raise AdmissionDeferred(
                "selected_profile_unobserved",
                "selected inference profile did not appear in the post-activation resource snapshot",
            )

    def _controllers_for(self, plan: ExecutionPlan) -> list[ControllerPolicy]:
        values = [
            item for item in self.policy.controllers
            if plan.service_class in item.throttle_for and item.profile_id != plan.profile_id
        ]
        return sorted(values, key=lambda item: (item.priority, item.id))

    def _restore_handle(self, handle: LeaseHandle) -> None:
        policies = {item.id: item for item in self.policy.controllers}
        for item in reversed(handle.throttled):
            controller = policies.get(str(item["controllerId"]))
            if controller is None:
                raise AdmissionDeferred("controller_missing", "recorded throttle controller is no longer configured")
            self.control_client.set_mode(
                controller,
                mode=str(item["normalMode"]),
                lease_id=handle.lease["id"],
                fencing_token=handle.lease["fencingToken"],
                broker_epoch=handle.lease["brokerEpoch"],
                generation=int(item["generation"]) + 1_000_000,
                reason=f"release:{handle.lease['jobId']}",
            )

    def _restore_recorded_throttles(self, lease: dict[str, Any]) -> None:
        self._restore_handle(LeaseHandle(lease=lease, throttled=list(lease.get("throttle", []))))

    def _recover_recorded_throttles(self, lease: dict[str, Any]) -> int:
        policies = {item.id: item for item in self.policy.controllers}
        recovery_fence = f"fence_{self.epoch}_recovery_{uuid.uuid4().hex}"
        expected_generation = 0
        for offset, item in enumerate(reversed(lease.get("throttle", [])), start=1):
            controller = policies.get(str(item["controllerId"]))
            if controller is None:
                raise AdmissionDeferred("controller_missing", "recorded throttle controller is no longer configured")
            acknowledgement = self.control_client.takeover_mode(
                controller,
                previous_lease_id=lease["id"],
                previous_fencing_token=lease["fencingToken"],
                recovery_fencing_token=recovery_fence,
                broker_epoch=self.epoch,
                generation=1_000_000 + offset,
                mode=str(item["normalMode"]),
            )
            expected_generation = max(expected_generation, int(acknowledgement["snapshotGeneration"]))
        return expected_generation

    def _combined_snapshot(self) -> dict[str, Any]:
        local = self.host_probe.snapshot()
        if not self.policy.probe_endpoint or not self.policy.probe_token_file:
            if self.policy.require_probe:
                raise AdmissionDeferred("resource_probe_required", "resource probe is required but not configured")
            return {**local, "probeHealthy": None, "unknownConsumers": None, "activeProfiles": [], "profiles": {}}
        try:
            remote = self.control_client.snapshot(self.policy.probe_endpoint, self.policy.probe_token_file)
        except AdmissionDeferred:
            if self.policy.require_probe:
                raise
            return {**local, "probeHealthy": False, "unknownConsumers": None, "activeProfiles": [], "profiles": {}}
        active_profiles = remote.get("activeProfiles", [])
        unknown = remote.get("unknownConsumers", 0)
        if not isinstance(active_profiles, list) or any(not isinstance(item, str) for item in active_profiles):
            raise AdmissionDeferred("resource_probe_invalid", "resource probe activeProfiles is invalid")
        if not isinstance(unknown, int) or isinstance(unknown, bool) or unknown < 0:
            raise AdmissionDeferred("resource_probe_invalid", "resource probe unknownConsumers is invalid")
        profiles = remote.get("profiles", {})
        if not isinstance(profiles, dict) or any(not isinstance(key, str) or not isinstance(value, dict) for key, value in profiles.items()):
            raise AdmissionDeferred("resource_probe_invalid", "resource probe profiles is invalid")
        return {
            **local,
            "probeHealthy": remote.get("health") == "healthy",
            "unknownConsumers": unknown,
            "activeProfiles": active_profiles,
            "profiles": profiles,
            "probeGeneration": remote.get("generation"),
        }

    def _validate_snapshot(self, snapshot: dict[str, Any], plan: ExecutionPlan, *, check_capacity: bool) -> None:
        if self.policy.require_probe and snapshot.get("probeHealthy") is not True:
            raise AdmissionDeferred("resource_probe_unhealthy", "resource probe did not report healthy")
        if snapshot.get("unknownConsumers") not in {None, 0}:
            raise AdmissionDeferred("unknown_gpu_consumer", "an unknown GPU consumer is present; admission is quarantined", retry_after_seconds=10)
        profile = snapshot.get("profiles", {}).get(plan.profile_id)
        if isinstance(profile, dict) and profile.get("health") != "healthy":
            raise AdmissionDeferred("selected_profile_unhealthy", "selected profile is not healthy in the admission snapshot")
        pressure = snapshot.get("memoryPressureAvg10")
        if pressure is not None and pressure > self.policy.maximum_memory_pressure_avg10:
            raise AdmissionDeferred("memory_pressure", "host memory pressure is above the admission threshold")
        if not check_capacity or not self.policy.enforce_memory_admission:
            return
        available_bytes = snapshot.get("availableMemoryBytes")
        if not isinstance(available_bytes, int):
            raise AdmissionDeferred("memory_unknown", "MemAvailable is unavailable")
        resident = plan.profile_id in snapshot.get("activeProfiles", [])
        required_gb = self.policy.host_reserve_gb + (0 if resident else plan.estimated_memory_gb)
        if available_bytes < int(required_gb * 1024**3):
            raise AdmissionDeferred(
                "insufficient_memory",
                "measured unified-memory headroom is below the configured envelope",
                detail={"requiredGb": required_gb, "availableBytes": available_bytes, "resident": resident},
            )

    @staticmethod
    def _released_profile_is_verified(
        snapshot: dict[str, Any], profile_id: str, *, requires_absence: bool
    ) -> bool:
        if snapshot.get("probeHealthy") is not True or snapshot.get("unknownConsumers") != 0:
            return False
        return not requires_absence or profile_id not in snapshot.get("activeProfiles", [])

    def preemption_candidate(self, plan: ExecutionPlan) -> bool:
        """Pure fail-closed check used before signalling cooperative yield."""
        if plan.service_class != "interactive" or plan.resource_group is None:
            return False
        try:
            snapshot = self._combined_snapshot()
            self._validate_snapshot(snapshot, plan, check_capacity=False)
        except AdmissionDeferred:
            return False
        total = snapshot.get("totalMemoryBytes")
        if isinstance(total, int):
            required = int((self.policy.host_reserve_gb + plan.estimated_memory_gb) * 1024**3)
            if required > total:
                return False
        return True

    def _validate_compatibility(self, snapshot: dict[str, Any], plan: ExecutionPlan) -> None:
        if snapshot.get("probeHealthy") is not True:
            return
        others = {
            profile for profile in snapshot.get("activeProfiles", [])
            if profile != plan.profile_id
        }
        if not others:
            return
        if plan.lease_mode == "exclusive":
            raise AdmissionDeferred("exclusive_profile_conflict", "another GPU profile remains active after lifecycle controls")
        certified = set(self.policy.shared_certifications)
        for profile in others:
            pair = tuple(sorted((profile, plan.profile_id)))
            if pair not in certified:
                raise AdmissionDeferred(
                    "uncertified_coexistence",
                    "inference and background work may share only under an exact certified profile pair",
                    detail={"profiles": list(pair)},
                )

    @staticmethod
    def _public_lease(lease: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": lease["id"], "jobId": lease["jobId"], "resourceGroup": lease["resourceGroup"],
            "profileId": lease["profileId"], "routeId": lease["routeId"], "serviceClass": lease["serviceClass"],
            "mode": lease["mode"], "status": lease["status"], "brokerEpoch": lease["brokerEpoch"],
        }
