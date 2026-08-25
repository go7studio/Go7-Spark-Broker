from __future__ import annotations

import uuid
from typing import Any

from spark_broker import PROTOCOL_VERSION


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
