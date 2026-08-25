from __future__ import annotations

import uuid
from typing import Any

from spark_broker import PROTOCOL_VERSION


class HealthyHostProbe:
    def snapshot(self) -> dict[str, Any]:
        return {
            "availableMemoryBytes": 120 * 1024**3,
            "totalMemoryBytes": 128 * 1024**3,
            "swapFreeBytes": 0,
            "memoryPressureAvg10": 0.0,
            "sampledAtMonotonic": 1.0,
        }


def write_resource_policy(root: Any, endpoint: str) -> Any:
    import json
    from pathlib import Path

    root = Path(root)
    token = root / "governor-token"
    token.write_text("g" * 32, encoding="utf-8")
    token.chmod(0o600)
    policy = root / "resource-policy.json"
    policy.write_text(json.dumps({
        "version": 1,
        "hostReserveGb": 4,
        "maximumMemoryPressureAvg10": 5,
        "enforceMemoryAdmission": True,
        "probe": {"endpoint": endpoint, "tokenFile": str(token), "required": True},
        "controllers": [],
        "sharedCertifications": [],
    }), encoding="utf-8")
    return policy


def request(**updates: Any) -> dict[str, Any]:
    suffix = uuid.uuid4().hex
    value: dict[str, Any] = {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": f"req_{suffix}",
        "traceId": f"trace_{suffix}",
        "idempotencyKey": f"idem_{suffix}",
        "origin": "tests",
        "visitedSystems": [],
        "hopCount": 0,
        "capability": "system.echo",
        "priority": 50,
        "inputs": [],
        "requiredOutputs": [],
        "constraints": {},
        "workflow": {"autoContinue": False, "approvedCapabilities": [], "maxContinuations": 0},
        "metadata": {},
    }
    value.update(updates)
    return value
