from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from . import PROTOCOL_VERSION


class ContractError(ValueError):
    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.field:
            value["field"] = self.field
        return value


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CAPABILITY = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9][a-z0-9]*){1,7}$")
_ARTIFACT_ID = re.compile(r"^art_[a-f0-9]{32}$")

TOP_LEVEL_FIELDS = {
    "protocolVersion",
    "requestId",
    "traceId",
    "idempotencyKey",
    "origin",
    "visitedSystems",
    "hopCount",
    "capability",
    "priority",
    "deadline",
    "inputs",
    "requiredOutputs",
    "constraints",
    "workflow",
    "metadata",
}

JOB_STATES = {
    "queued",
    "loading",
    "running",
    "validating",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("invalid_type", f"{field} must be an object", field=field)
    return value


def _string(value: Any, field: str, *, maximum: int = 4096, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ContractError("invalid_string", f"{field} must be a non-empty string up to {maximum} characters", field=field)
    if pattern and not pattern.fullmatch(value):
        raise ContractError("invalid_format", f"{field} has an invalid format", field=field)
    return value


def _json_size(value: Any, field: str, maximum: int) -> None:
    try:
        size = len(canonical_json(value).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_json", f"{field} must contain JSON values", field=field) from exc
    if size > maximum:
        raise ContractError("too_large", f"{field} exceeds {maximum} bytes", field=field)


def _validate_deadline(value: Any) -> str | None:
    if value is None:
        return None
    text = _string(value, "deadline", maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("invalid_deadline", "deadline must be RFC 3339", field="deadline") from exc
    if parsed.tzinfo is None:
        raise ContractError("invalid_deadline", "deadline must include a timezone", field="deadline")
    if parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ContractError("deadline_expired", "deadline is already expired", field="deadline")
    return text


def validate_artifact_reference(value: Any, field: str) -> dict[str, Any]:
    item = _object(value, field)
    unknown = set(item) - {"artifactId", "role", "sha256"}
    if unknown:
        raise ContractError("unknown_field", f"{field} contains unknown fields: {sorted(unknown)}", field=field)
    artifact_id = _string(item.get("artifactId"), f"{field}.artifactId", maximum=36, pattern=_ARTIFACT_ID)
    result: dict[str, Any] = {"artifactId": artifact_id}
    if "role" in item:
        result["role"] = _string(item["role"], f"{field}.role", maximum=64, pattern=_ID)
    if "sha256" in item:
        sha = _string(item["sha256"], f"{field}.sha256", maximum=64)
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ContractError("invalid_format", f"{field}.sha256 must be lowercase hex", field=f"{field}.sha256")
        result["sha256"] = sha
    return result


def validate_required_output(value: Any, field: str) -> dict[str, Any]:
    item = _object(value, field)
    unknown = set(item) - {"role", "kind", "mediaTypes", "required"}
    if unknown:
        raise ContractError("unknown_field", f"{field} contains unknown fields: {sorted(unknown)}", field=field)
    role = _string(item.get("role"), f"{field}.role", maximum=64, pattern=_ID)
    kind = _string(item.get("kind"), f"{field}.kind", maximum=64, pattern=_ID)
    media_types = item.get("mediaTypes", [])
    if not isinstance(media_types, list) or not media_types or len(media_types) > 8:
        raise ContractError("invalid_output", f"{field}.mediaTypes must contain 1-8 values", field=f"{field}.mediaTypes")
    clean_media = [_string(x, f"{field}.mediaTypes", maximum=128) for x in media_types]
    required = item.get("required", True)
    if not isinstance(required, bool):
        raise ContractError("invalid_type", f"{field}.required must be boolean", field=f"{field}.required")
    return {"role": role, "kind": kind, "mediaTypes": clean_media, "required": required}


def validate_job_request(value: Any, *, broker_id: str, max_hops: int = 8) -> dict[str, Any]:
    body = _object(value, "request")
    unknown = set(body) - TOP_LEVEL_FIELDS
    if unknown:
        raise ContractError("unknown_field", f"request contains unknown fields: {sorted(unknown)}")
    missing = {"protocolVersion", "requestId", "traceId", "idempotencyKey", "origin", "capability"} - set(body)
    if missing:
        raise ContractError("missing_field", f"request is missing fields: {sorted(missing)}")
    if body["protocolVersion"] != PROTOCOL_VERSION:
        raise ContractError("unsupported_protocol", f"only protocolVersion {PROTOCOL_VERSION} is supported", field="protocolVersion")

    visited = body.get("visitedSystems", [])
    if not isinstance(visited, list) or len(visited) > max_hops:
        raise ContractError("invalid_route", f"visitedSystems must contain at most {max_hops} entries", field="visitedSystems")
    visited = [_string(x, "visitedSystems[]", maximum=128, pattern=_ID) for x in visited]
    if len(visited) != len(set(visited)):
        raise ContractError("route_cycle", "visitedSystems contains a duplicate", field="visitedSystems")
    if broker_id in visited:
        raise ContractError("route_cycle", "request already visited this broker", field="visitedSystems")
    hop_count = body.get("hopCount", 0)
    if not isinstance(hop_count, int) or isinstance(hop_count, bool) or hop_count < 0 or hop_count >= max_hops:
        raise ContractError("hop_limit", f"hopCount must be between 0 and {max_hops - 1}", field="hopCount")
    if hop_count != len(visited):
        raise ContractError("invalid_route", "hopCount must equal visitedSystems length", field="hopCount")

    priority = body.get("priority", 50)
    if not isinstance(priority, int) or isinstance(priority, bool) or not 0 <= priority <= 100:
        raise ContractError("invalid_priority", "priority must be an integer from 0 through 100", field="priority")

    inputs = body.get("inputs", [])
    if not isinstance(inputs, list) or len(inputs) > 32:
        raise ContractError("invalid_inputs", "inputs must contain at most 32 artifact references", field="inputs")
    clean_inputs = [validate_artifact_reference(x, f"inputs[{i}]") for i, x in enumerate(inputs)]

    outputs = body.get("requiredOutputs", [])
    if not isinstance(outputs, list) or len(outputs) > 32:
        raise ContractError("invalid_outputs", "requiredOutputs must contain at most 32 entries", field="requiredOutputs")
    clean_outputs = [validate_required_output(x, f"requiredOutputs[{i}]") for i, x in enumerate(outputs)]
    roles = [x["role"] for x in clean_outputs]
    if len(roles) != len(set(roles)):
        raise ContractError("duplicate_output", "required output roles must be unique", field="requiredOutputs")

    constraints = body.get("constraints", {})
    metadata = body.get("metadata", {})
    workflow = body.get("workflow", {})
    for field, item, maximum in (
        ("constraints", constraints, 32_768),
        ("metadata", metadata, 16_384),
        ("workflow", workflow, 16_384),
    ):
        _object(item, field)
        _json_size(item, field, maximum)

    allowed_workflow = {"autoContinue", "approvedCapabilities", "maxContinuations"}
    if set(workflow) - allowed_workflow:
        raise ContractError("unknown_field", f"workflow contains unknown fields: {sorted(set(workflow) - allowed_workflow)}", field="workflow")
    auto_continue = workflow.get("autoContinue", False)
    if not isinstance(auto_continue, bool):
        raise ContractError("invalid_type", "workflow.autoContinue must be boolean", field="workflow.autoContinue")
    approved = workflow.get("approvedCapabilities", [])
    if not isinstance(approved, list) or len(approved) > 16:
        raise ContractError("invalid_workflow", "workflow.approvedCapabilities must contain at most 16 values", field="workflow.approvedCapabilities")
    approved = [_string(x, "workflow.approvedCapabilities[]", maximum=128, pattern=_CAPABILITY) for x in approved]
    max_continuations = workflow.get("maxContinuations", 0)
    if not isinstance(max_continuations, int) or isinstance(max_continuations, bool) or not 0 <= max_continuations <= 16:
        raise ContractError("invalid_workflow", "workflow.maxContinuations must be 0-16", field="workflow.maxContinuations")

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": _string(body["requestId"], "requestId", maximum=128, pattern=_ID),
        "traceId": _string(body["traceId"], "traceId", maximum=128, pattern=_ID),
        "idempotencyKey": _string(body["idempotencyKey"], "idempotencyKey", maximum=128, pattern=_ID),
        "origin": _string(body["origin"], "origin", maximum=128, pattern=_ID),
        "visitedSystems": visited,
        "hopCount": hop_count,
        "capability": _string(body["capability"], "capability", maximum=128, pattern=_CAPABILITY),
        "priority": priority,
        "deadline": _validate_deadline(body.get("deadline")),
        "inputs": clean_inputs,
        "requiredOutputs": clean_outputs,
        "constraints": constraints,
        "workflow": {
            "autoContinue": auto_continue,
            "approvedCapabilities": approved,
            "maxContinuations": max_continuations,
        },
        "metadata": metadata,
    }


def validate_upload_metadata(kind: str, role: str, media_type: str) -> tuple[str, str, str]:
    return (
        _string(kind, "kind", maximum=64, pattern=_ID),
        _string(role, "role", maximum=64, pattern=_ID),
        _string(media_type, "mediaType", maximum=128),
    )
