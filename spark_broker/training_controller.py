from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import socket
import stat
import subprocess
import threading
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from .secure_files import SecureFileError, read_owner_secret, read_owner_text


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
_SYSTEMD_UNIT = re.compile(r"^[A-Za-z0-9:_.@\-]{1,240}\.service$")
_MUTATION_ID = re.compile(r"^mutation_[a-f0-9]{32}$")
_LEASE_ID = re.compile(r"^lease_[a-f0-9]{32}$")
_FENCE = re.compile(r"^fence_[A-Za-z0-9_]{1,240}$")
_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class TrainingControllerError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class TrainingControllerPolicy:
    controller_id: str
    profile_id: str
    unit: str
    normal_mode: str
    released_mode: str
    checkpoint_root: Path
    checkpoint_receipt_file: Path
    state_file: Path
    authority_file: Path
    allow_fresh_start: bool = False
    stop_timeout_seconds: int = 600

    @classmethod
    def from_file(cls, path: Path) -> "TrainingControllerPolicy":
        try:
            value = json.loads(
                read_owner_text(path, "training controller config", maximum_bytes=1024 * 1024)
            )
        except SecureFileError as exc:
            raise ValueError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise ValueError("training controller config must be UTF-8 JSON") from exc
        required = {
            "version", "controllerId", "profileId", "unit", "normalMode",
            "releasedMode", "checkpointRoot", "checkpointReceiptFile",
            "stateFile", "authorityFile",
        }
        optional = {"allowFreshStart", "stopTimeoutSeconds"}
        if not isinstance(value, dict) or not required.issubset(value) or set(value) - (required | optional):
            raise ValueError("training controller config has an invalid schema")
        if value["version"] != 1:
            raise ValueError("training controller config version must be 1")

        def identifier(name: str) -> str:
            item = value[name]
            if not isinstance(item, str) or not _IDENTIFIER.fullmatch(item):
                raise ValueError(f"{name} is invalid")
            return item

        unit = value["unit"]
        if not isinstance(unit, str) or not _SYSTEMD_UNIT.fullmatch(unit):
            raise ValueError("unit is invalid")

        def absolute(name: str) -> Path:
            item = value[name]
            if not isinstance(item, str) or not Path(item).is_absolute():
                raise ValueError(f"{name} must be an absolute path")
            return Path(item)

        root = absolute("checkpointRoot").resolve()
        receipt = absolute("checkpointReceiptFile")
        try:
            receipt.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("checkpointReceiptFile must be inside checkpointRoot") from exc
        state_file = absolute("stateFile")
        authority_file = absolute("authorityFile")
        if state_file == authority_file:
            raise ValueError("stateFile and authorityFile must differ")
        if identifier("normalMode") == identifier("releasedMode"):
            raise ValueError("normalMode and releasedMode must differ")
        allow_fresh = value.get("allowFreshStart", False)
        timeout = value.get("stopTimeoutSeconds", 600)
        if not isinstance(allow_fresh, bool):
            raise ValueError("allowFreshStart must be boolean")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 3600:
            raise ValueError("stopTimeoutSeconds must be 1-3600")
        return cls(
            controller_id=identifier("controllerId"),
            profile_id=identifier("profileId"),
            unit=unit,
            normal_mode=identifier("normalMode"),
            released_mode=identifier("releasedMode"),
            checkpoint_root=root,
            checkpoint_receipt_file=receipt,
            state_file=state_file,
            authority_file=authority_file,
            allow_fresh_start=allow_fresh,
            stop_timeout_seconds=timeout,
        )


class UnitControl(Protocol):
    def start(self, unit: str, timeout: int) -> None: ...
    def stop(self, unit: str, timeout: int) -> None: ...
    def active(self, unit: str) -> bool: ...


class SystemdUserControl:
    @staticmethod
    def _environment() -> dict[str, str]:
        runtime = f"/run/user/{os.geteuid()}"
        value = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LC_ALL": "C",
        }
        if Path(runtime).is_dir():
            value["XDG_RUNTIME_DIR"] = runtime
            value["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime}/bus"
        return value

    def _run(self, argv: list[str], timeout: int) -> str:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
            env=self._environment(),
        )
        if result.returncode != 0:
            detail = result.stderr.strip()[:256] or f"exit {result.returncode}"
            raise TrainingControllerError("unit_control_failed", f"systemctl failed: {detail}", status=503)
        return result.stdout

    def start(self, unit: str, timeout: int) -> None:
        self._run(["systemctl", "--user", "start", unit], timeout)

    def stop(self, unit: str, timeout: int) -> None:
        self._run(["systemctl", "--user", "stop", unit], timeout)

    def active(self, unit: str) -> bool:
        value = self._run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            30,
        ).strip()
        if value not in {"active", "inactive", "failed", "activating", "deactivating"}:
            raise TrainingControllerError("unit_state_invalid", "systemd returned an invalid state", status=503)
        return value == "active"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.parent.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise TrainingControllerError(
            "state_directory_unsafe",
            "controller state directory must be service-owned and not group/world writable",
            status=503,
        )
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        raw = _canonical(value) + b"\n"
        os.write(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise TrainingControllerError("state_write_failed", "controller state could not be persisted", status=503) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _owner_json(path: Path, label: str, maximum_bytes: int = 1024 * 1024) -> dict[str, Any]:
    try:
        value = json.loads(read_owner_text(path, label, maximum_bytes=maximum_bytes))
    except (SecureFileError, json.JSONDecodeError) as exc:
        raise TrainingControllerError("state_invalid", f"{label} is invalid", status=503) from exc
    if not isinstance(value, dict):
        raise TrainingControllerError("state_invalid", f"{label} must be an object", status=503)
    return value


class TrainingController:
    def __init__(self, policy: TrainingControllerPolicy, *, units: UnitControl | None = None) -> None:
        self.policy = policy
        self.units = units or SystemdUserControl()
        self._lock = threading.Lock()
        with self._lock:
            self._recover_if_needed()

    def _authority(self) -> dict[str, Any] | None:
        if not self.policy.authority_file.exists():
            return None
        return _owner_json(self.policy.authority_file, "training controller authority")

    def _checkpoint_path(self, relative: str) -> Path:
        candidate = self.policy.checkpoint_root / relative
        current = self.policy.checkpoint_root
        try:
            for part in Path(relative).parts:
                current = current / part
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise TrainingControllerError(
                        "checkpoint_invalid",
                        "checkpoint paths cannot contain symbolic links",
                        status=503,
                    )
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.policy.checkpoint_root)
        except TrainingControllerError:
            raise
        except (OSError, ValueError) as exc:
            raise TrainingControllerError(
                "checkpoint_invalid",
                "checkpoint file is outside the configured root",
                status=503,
            ) from exc
        return resolved

    def _load_receipt(self) -> dict[str, Any]:
        receipt = _owner_json(
            self.policy.checkpoint_receipt_file,
            "checkpoint receipt",
            maximum_bytes=4 * 1024 * 1024,
        )
        if set(receipt) != {"version", "runId", "checkpointId", "files"} or receipt.get("version") != 1:
            raise TrainingControllerError("checkpoint_invalid", "checkpoint receipt schema is invalid", status=503)
        run_id = receipt.get("runId")
        checkpoint_id = receipt.get("checkpointId")
        files = receipt.get("files")
        if (
            not isinstance(run_id, str)
            or not _CHECKPOINT_ID.fullmatch(run_id)
            or not isinstance(checkpoint_id, str)
            or not _CHECKPOINT_ID.fullmatch(checkpoint_id)
            or not isinstance(files, list)
            or not 1 <= len(files) <= 128
        ):
            raise TrainingControllerError("checkpoint_invalid", "checkpoint receipt identity is invalid", status=503)
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise TrainingControllerError("checkpoint_invalid", "checkpoint file record is invalid", status=503)
            relative = item.get("path")
            size = item.get("size")
            digest = item.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in seen
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                raise TrainingControllerError("checkpoint_invalid", "checkpoint file record is invalid", status=503)
            seen.add(relative)
        return receipt

    @staticmethod
    def _receipt_content_digest(receipt: dict[str, Any]) -> str:
        content = {
            "runId": receipt["runId"],
            "files": sorted(receipt["files"], key=lambda item: item["path"]),
        }
        return hashlib.sha256(_canonical(content)).hexdigest()

    def _receipt_identity_if_present(self) -> dict[str, str] | None:
        try:
            metadata = self.policy.checkpoint_receipt_file.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise TrainingControllerError(
                "checkpoint_invalid",
                "checkpoint receipt cannot be inspected safely",
                status=503,
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise TrainingControllerError(
                "checkpoint_invalid",
                "checkpoint receipt must be a regular file",
                status=503,
            )
        receipt = self._load_receipt()
        return {
            "runId": receipt["runId"],
            "checkpointId": receipt["checkpointId"],
            "contentSha256": self._receipt_content_digest(receipt),
        }

    @staticmethod
    def _validate_release_baseline(value: Any) -> dict[str, str] | None:
        if value is None:
            return None
        if (
            not isinstance(value, dict)
            or set(value) != {"runId", "checkpointId", "contentSha256"}
            or not isinstance(value.get("runId"), str)
            or not _CHECKPOINT_ID.fullmatch(value["runId"])
            or not isinstance(value.get("checkpointId"), str)
            or not _CHECKPOINT_ID.fullmatch(value["checkpointId"])
            or not isinstance(value.get("contentSha256"), str)
            or not _SHA256.fullmatch(value["contentSha256"])
        ):
            raise TrainingControllerError(
                "recovery_required",
                "training controller release baseline is invalid",
                status=503,
            )
        return value

    def _checkpoint(self) -> tuple[dict[str, str], dict[str, Any]]:
        receipt = self._load_receipt()
        run_id = receipt["runId"]
        checkpoint_id = receipt["checkpointId"]
        normalized: list[dict[str, Any]] = []
        for item in receipt["files"]:
            relative = item["path"]
            size = item["size"]
            digest = item["sha256"]
            path = self._checkpoint_path(relative)
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags)
            except OSError as exc:
                raise TrainingControllerError("checkpoint_invalid", "checkpoint file cannot be opened safely", status=503) from exc
            observed = hashlib.sha256()
            observed_size = 0
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise TrainingControllerError("checkpoint_invalid", "checkpoint file is not regular", status=503)
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    observed.update(chunk)
                    observed_size += len(chunk)
            finally:
                os.close(descriptor)
            if observed_size != size or not hmac.compare_digest(observed.hexdigest(), digest):
                raise TrainingControllerError("checkpoint_changed", "checkpoint content differs from its receipt", status=503)
            normalized.append({"path": relative, "size": size, "sha256": digest})
        evidence = {
            "version": 1,
            "runId": run_id,
            "checkpointId": checkpoint_id,
            "files": sorted(normalized, key=lambda item: item["path"]),
        }
        digest = hashlib.sha256(_canonical(evidence)).hexdigest()
        return {"runId": run_id, "checkpointId": checkpoint_id, "sha256": digest}, evidence

    def _validate_request(self, value: Any, *, takeover: bool) -> dict[str, Any]:
        required = (
            {
                "protocolVersion", "mutationId", "previousLeaseId",
                "previousFencingToken", "recoveryFencingToken", "brokerEpoch",
                "controlGeneration", "targetMode", "reason",
            }
            if takeover
            else {
                "protocolVersion", "mutationId", "leaseId", "fencingToken",
                "brokerEpoch", "controlGeneration", "targetMode", "reason",
            }
        )
        if not isinstance(value, dict) or set(value) != required:
            raise TrainingControllerError("invalid_request", "controller request schema is invalid", status=400)
        lease_key = "previousLeaseId" if takeover else "leaseId"
        fence_keys = ("previousFencingToken", "recoveryFencingToken") if takeover else ("fencingToken",)
        if (
            value.get("protocolVersion") != "1.0"
            or not isinstance(value.get("mutationId"), str)
            or not _MUTATION_ID.fullmatch(value["mutationId"])
            or not isinstance(value.get(lease_key), str)
            or not _LEASE_ID.fullmatch(value[lease_key])
            or any(not isinstance(value.get(key), str) or not _FENCE.fullmatch(value[key]) for key in fence_keys)
            or not isinstance(value.get("brokerEpoch"), int)
            or isinstance(value.get("brokerEpoch"), bool)
            or value["brokerEpoch"] < 1
            or not isinstance(value.get("controlGeneration"), int)
            or isinstance(value.get("controlGeneration"), bool)
            or value["controlGeneration"] < 1
            or value.get("targetMode") not in {self.policy.normal_mode, self.policy.released_mode}
            or not isinstance(value.get("reason"), str)
            or not 1 <= len(value["reason"]) <= 512
        ):
            raise TrainingControllerError("invalid_request", "controller request fields are invalid", status=400)
        return value

    @staticmethod
    def _public_from_authority(authority: dict[str, Any]) -> dict[str, Any]:
        public = authority.get("public")
        if not isinstance(public, dict):
            raise TrainingControllerError("state_invalid", "controller authority is incomplete", status=503)
        return public

    def _check_fence(self, request: dict[str, Any], current: dict[str, Any] | None) -> None:
        if current is None:
            return
        public = self._public_from_authority(current)
        epoch = request["brokerEpoch"]
        if epoch < public["brokerEpoch"]:
            raise TrainingControllerError("stale_epoch", "broker epoch is stale")
        if epoch == public["brokerEpoch"]:
            if request["leaseId"] == public["leaseId"]:
                if request["fencingToken"] != public["fencingToken"]:
                    raise TrainingControllerError("fence_conflict", "lease fence differs")
                if request["controlGeneration"] <= public["controlGeneration"]:
                    raise TrainingControllerError("generation_conflict", "control generation did not advance")
            elif public["effectiveMode"] != self.policy.normal_mode or request["controlGeneration"] != 1:
                raise TrainingControllerError("lease_conflict", "another lease is still authoritative")
        elif public["effectiveMode"] != self.policy.normal_mode:
            raise TrainingControllerError("takeover_required", "new epoch must take over the displaced state")

    def _apply(
        self,
        request: dict[str, Any],
        current: dict[str, Any] | None,
        *,
        release_baseline: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        target = request["targetMode"]
        checkpoint: dict[str, str] | None = None
        evidence: dict[str, Any] | None = None
        if target == self.policy.released_mode:
            self.units.stop(self.policy.unit, self.policy.stop_timeout_seconds)
            if self.units.active(self.policy.unit):
                raise TrainingControllerError("release_unverified", "training unit remains active", status=503)
            checkpoint, evidence = self._checkpoint()
            if release_baseline is not None and (
                checkpoint["runId"] != release_baseline["runId"]
                or checkpoint["checkpointId"] == release_baseline["checkpointId"]
                or self._receipt_content_digest(evidence)
                == release_baseline["contentSha256"]
            ):
                raise TrainingControllerError(
                    "checkpoint_not_advanced",
                    "training stop did not publish a new immutable checkpoint",
                    status=503,
                )
        else:
            if current and isinstance(current.get("checkpointEvidence"), dict):
                checkpoint, evidence = self._checkpoint()
                expected = current["checkpointEvidence"]
                if evidence != expected:
                    raise TrainingControllerError("checkpoint_changed", "resume checkpoint evidence changed", status=503)
            elif not self.policy.allow_fresh_start:
                raise TrainingControllerError("checkpoint_required", "fresh training start is not permitted", status=503)
            self.units.start(self.policy.unit, self.policy.stop_timeout_seconds)
            if not self.units.active(self.policy.unit):
                raise TrainingControllerError("resume_unverified", "training unit did not become active", status=503)
        return checkpoint or {}, evidence

    def _complete_set_mode(
        self,
        request: dict[str, Any],
        previous: dict[str, Any] | None,
        release_baseline: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        checkpoint, evidence = self._apply(
            request,
            previous,
            release_baseline=release_baseline,
        )
        public: dict[str, Any] = {
            "protocolVersion": "1.0",
            "controllerId": self.policy.controller_id,
            "mutationId": request["mutationId"],
            "leaseId": request["leaseId"],
            "fencingToken": request["fencingToken"],
            "brokerEpoch": request["brokerEpoch"],
            "controlGeneration": request["controlGeneration"],
            "effectiveMode": request["targetMode"],
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        response: dict[str, Any] = {
            "mutationId": request["mutationId"],
            "leaseId": request["leaseId"],
            "fencingToken": request["fencingToken"],
            "brokerEpoch": request["brokerEpoch"],
            "acknowledgedGeneration": request["controlGeneration"],
            "effectiveMode": request["targetMode"],
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        if checkpoint:
            public["checkpoint"] = checkpoint
            response["checkpoint"] = checkpoint
        authority = {
            "version": 1,
            "phase": "ready",
            "request": request,
            "public": public,
            "response": response,
            "checkpointEvidence": evidence,
        }
        _atomic_json(self.policy.authority_file, authority)
        _atomic_json(self.policy.state_file, public)
        return response

    def _complete_takeover(
        self,
        request: dict[str, Any],
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        synthetic = {
            "protocolVersion": "1.0",
            "mutationId": request["mutationId"],
            "leaseId": request["previousLeaseId"],
            "fencingToken": request["recoveryFencingToken"],
            "brokerEpoch": request["brokerEpoch"],
            "controlGeneration": request["controlGeneration"],
            "targetMode": request["targetMode"],
            "reason": request["reason"],
        }
        _checkpoint, evidence = self._apply(synthetic, previous)
        public = {
            "protocolVersion": "1.0",
            "controllerId": self.policy.controller_id,
            "mutationId": request["mutationId"],
            "leaseId": request["previousLeaseId"],
            "fencingToken": request["recoveryFencingToken"],
            "brokerEpoch": request["brokerEpoch"],
            "controlGeneration": request["controlGeneration"],
            "effectiveMode": request["targetMode"],
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        response = {
            "mutationId": request["mutationId"],
            "previousLeaseId": request["previousLeaseId"],
            "previousFencingToken": request["previousFencingToken"],
            "recoveryFencingToken": request["recoveryFencingToken"],
            "brokerEpoch": request["brokerEpoch"],
            "acknowledgedGeneration": request["controlGeneration"],
            "effectiveMode": request["targetMode"],
            "health": "healthy",
            "appliedAtSafeBoundary": True,
        }
        authority = {
            "version": 1,
            "phase": "ready",
            "takeoverRequest": request,
            "public": public,
            "response": response,
            "checkpointEvidence": evidence,
        }
        _atomic_json(self.policy.authority_file, authority)
        _atomic_json(self.policy.state_file, public)
        return response

    def _recover_if_needed(self) -> dict[str, Any] | None:
        authority = self._authority()
        if authority is None or authority.get("phase") == "ready":
            return authority
        if authority.get("version") != 1 or authority.get("phase") != "applying":
            raise TrainingControllerError(
                "recovery_required",
                "training controller authority cannot be recovered",
                status=503,
            )
        previous = authority.get("previous")
        if previous is not None and not isinstance(previous, dict):
            raise TrainingControllerError(
                "recovery_required",
                "training controller previous authority is invalid",
                status=503,
            )
        request = authority.get("request")
        takeover_request = authority.get("takeoverRequest")
        if request is not None and takeover_request is None:
            validated = self._validate_request(request, takeover=False)
            baseline = None
            if validated["targetMode"] == self.policy.released_mode:
                if "releaseBaseline" not in authority:
                    raise TrainingControllerError(
                        "recovery_required",
                        "training controller release baseline is missing",
                        status=503,
                    )
                baseline = self._validate_release_baseline(authority["releaseBaseline"])
            self._complete_set_mode(validated, previous, baseline)
        elif takeover_request is not None and request is None and isinstance(previous, dict):
            validated = self._validate_request(takeover_request, takeover=True)
            if validated["targetMode"] != self.policy.normal_mode:
                raise TrainingControllerError(
                    "recovery_required",
                    "training controller takeover recovery target is invalid",
                    status=503,
                )
            self._complete_takeover(validated, previous)
        else:
            raise TrainingControllerError(
                "recovery_required",
                "training controller applying authority is incomplete",
                status=503,
            )
        return self._authority()

    def set_mode(self, raw: Any) -> dict[str, Any]:
        request = self._validate_request(raw, takeover=False)
        with self._lock:
            current = self._recover_if_needed()
            if current and current.get("request") == request and current.get("phase") == "ready":
                public = self._public_from_authority(current)
                _atomic_json(self.policy.state_file, public)
                return dict(current["response"])
            self._check_fence(request, current)
            applying: dict[str, Any] = {
                "version": 1, "phase": "applying", "request": request,
                "previous": current,
            }
            baseline = None
            if request["targetMode"] == self.policy.released_mode:
                baseline = self._receipt_identity_if_present()
                if (
                    baseline is None
                    and current is not None
                    and isinstance(current.get("checkpointEvidence"), dict)
                ):
                    raise TrainingControllerError(
                        "checkpoint_required",
                        "the prior checkpoint receipt is unavailable",
                        status=503,
                    )
                applying["releaseBaseline"] = baseline
            _atomic_json(self.policy.authority_file, applying)
            return self._complete_set_mode(request, current, baseline)

    def takeover(self, raw: Any) -> dict[str, Any]:
        request = self._validate_request(raw, takeover=True)
        if request["targetMode"] != self.policy.normal_mode:
            raise TrainingControllerError("invalid_request", "takeover target must be normal mode", status=400)
        with self._lock:
            current = self._recover_if_needed()
            if current and current.get("takeoverRequest") == request and current.get("phase") == "ready":
                _atomic_json(self.policy.state_file, self._public_from_authority(current))
                return dict(current["response"])
            if current is None:
                raise TrainingControllerError("takeover_conflict", "there is no prior controller state")
            prior = self._public_from_authority(current)
            if (
                prior["leaseId"] != request["previousLeaseId"]
                or prior["fencingToken"] != request["previousFencingToken"]
                or request["brokerEpoch"] <= prior["brokerEpoch"]
            ):
                raise TrainingControllerError("takeover_conflict", "previous lease fence or epoch differs")
            _atomic_json(self.policy.authority_file, {
                "version": 1, "phase": "applying", "takeoverRequest": request,
                "previous": current,
            })
            return self._complete_takeover(request, current)


@dataclass(frozen=True)
class TrainingControllerServerConfig:
    bind: str
    port: int
    token: str

    def __post_init__(self) -> None:
        try:
            address = ipaddress.ip_address(self.bind)
        except ValueError as exc:
            raise ValueError("controller bind must be a numeric loopback address") from exc
        if not address.is_loopback or not 1 <= self.port <= 65535:
            raise ValueError("controller must use a valid loopback port")
        if not 32 <= len(self.token) <= 2048 or "\n" in self.token or "\r" in self.token:
            raise ValueError("controller token must contain 32-2048 characters")


class TrainingControllerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: TrainingControllerServerConfig, controller: TrainingController) -> None:
        self.config = config
        self.controller = controller
        if ":" in config.bind:
            self.address_family = socket.AF_INET6
        super().__init__((config.bind, config.port), TrainingControllerHandler)


class TrainingControllerHandler(BaseHTTPRequestHandler):
    server: TrainingControllerHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"spark-training-controller {self.client_address[0]} {fmt % args}", flush=True)

    def do_POST(self) -> None:
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
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if not 0 <= length <= 1024 * 1024:
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "invalid_body"}})
            return
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": {"code": "invalid_json"}})
            return
        try:
            if self.path == "/v1/resource-mode":
                response = self.server.controller.set_mode(value)
            elif self.path == "/v1/resource-takeover":
                response = self.server.controller.takeover(value)
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found"}})
                return
        except TrainingControllerError as exc:
            self._json(HTTPStatus(exc.status), {"error": {"code": exc.code, "message": str(exc)}})
            return
        self._json(HTTPStatus.OK, response)

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        raw = _canonical(value)
        self.close_connection = True
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    config_path = Path(os.environ.get("SPARK_TRAINING_CONTROLLER_CONFIG", ""))
    token_path = Path(os.environ.get("SPARK_TRAINING_CONTROLLER_TOKEN_FILE", ""))
    if not str(config_path) or not str(token_path):
        raise SystemExit("SPARK_TRAINING_CONTROLLER_CONFIG and SPARK_TRAINING_CONTROLLER_TOKEN_FILE are required")
    try:
        policy = TrainingControllerPolicy.from_file(config_path)
        token = read_owner_secret(token_path, "training controller credential")
        config = TrainingControllerServerConfig(
            bind=os.environ.get("SPARK_TRAINING_CONTROLLER_BIND", "127.0.0.1"),
            port=int(os.environ.get("SPARK_TRAINING_CONTROLLER_PORT", "9001")),
            token=token,
        )
    except (SecureFileError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    server = TrainingControllerHTTPServer(config, TrainingController(policy))
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
