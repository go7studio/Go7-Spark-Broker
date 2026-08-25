from __future__ import annotations

import base64
import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Any, Callable

from . import BROKER_VERSION, PROTOCOL_VERSION
from .cli import request_envelope
from .client import BrokerClient, ClientError


def client() -> BrokerClient:
    token = os.environ.get("SPARK_BROKER_TOKEN", "")
    token_file = os.environ.get("SPARK_BROKER_TOKEN_FILE", "")
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("SPARK_BROKER_TOKEN or SPARK_BROKER_TOKEN_FILE is required")
    return BrokerClient(os.environ.get("SPARK_BROKER_URL", "http://127.0.0.1:8790"), token, timeout=int(os.environ.get("SPARK_MCP_TIMEOUT", "60")))


TOOLS: list[dict[str, Any]] = [
    {"name": "spark_capabilities", "description": "List installed typed capabilities and broker limits.", "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "spark_submit_job", "description": "Submit a complete protocol v1 asynchronous capability request.", "inputSchema": {"type": "object", "properties": {"request": {"type": "object"}}, "required": ["request"], "additionalProperties": False}},
    {"name": "spark_generate_3d", "description": "Submit a typed 3D generation job using an uploaded image artifact.", "inputSchema": {"type": "object", "properties": {
        "artifactId": {"type": "string"}, "artifactSha256": {"type": "string"}, "origin": {"type": "string"}, "idempotencyKey": {"type": "string"},
        "mode": {"enum": ["shape", "pbr"]}, "priority": {"type": "integer", "minimum": 0, "maximum": 100}, "seed": {"type": "integer"},
        "maxFaces": {"type": "integer"}, "targetEngine": {"enum": ["generic", "blender", "godot", "unity", "unreal"]},
        "requireWatertight": {"type": "boolean"}, "authorizeBlenderContinuation": {"type": "boolean"}, "autoContinue": {"type": "boolean"}
    }, "required": ["artifactId", "artifactSha256", "origin", "idempotencyKey"], "additionalProperties": False}},
    {"name": "spark_upload_artifact", "description": "Upload a base64-encoded input artifact (including GLB models) to the broker registry.", "inputSchema": {"type": "object", "properties": {
        "base64": {"type": "string"}, "kind": {"type": "string"}, "role": {"type": "string"}, "mediaType": {"type": "string"}, "origin": {"type": "string"}
    }, "required": ["base64", "kind", "role", "mediaType", "origin"], "additionalProperties": False}},
    {"name": "spark_chat", "description": "Submit typed local-model text generation using an uploaded prompt artifact.", "inputSchema": {"type": "object", "properties": {
        "artifactId": {"type": "string"}, "artifactSha256": {"type": "string"}, "origin": {"type": "string"}, "idempotencyKey": {"type": "string"},
        "priority": {"type": "integer", "minimum": 0, "maximum": 100}, "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "maxTokens": {"type": "integer", "minimum": 1, "maximum": 32768}, "systemPrompt": {"type": "string"}, "timeoutSeconds": {"type": "integer", "minimum": 10, "maximum": 1800}, "enableThinking": {"type": "boolean"},
        "model": {"type": "string"}, "serviceClass": {"enum": ["interactive", "batch", "background"]},
        "routePreference": {"enum": ["balanced", "latency", "throughput", "memory"]}
    }, "required": ["artifactId", "artifactSha256", "origin", "idempotencyKey"], "additionalProperties": False}},
    {"name": "spark_job_status", "description": "Read a durable asynchronous job and its typed result.", "inputSchema": {"type": "object", "properties": {"jobId": {"type": "string"}}, "required": ["jobId"], "additionalProperties": False}},
    {"name": "spark_job_events", "description": "Read the persisted state transition journal for a job.", "inputSchema": {"type": "object", "properties": {"jobId": {"type": "string"}}, "required": ["jobId"], "additionalProperties": False}},
    {"name": "spark_cancel_job", "description": "Request cancellation of a queued or active job.", "inputSchema": {"type": "object", "properties": {"jobId": {"type": "string"}}, "required": ["jobId"], "additionalProperties": False}},
    {"name": "spark_artifact_metadata", "description": "Read artifact type, role, hash, size, relationships, and validation metadata.", "inputSchema": {"type": "object", "properties": {"artifactId": {"type": "string"}}, "required": ["artifactId"], "additionalProperties": False}},
    {"name": "spark_download_artifact", "description": "Download a small artifact as base64 after hash verification.", "inputSchema": {"type": "object", "properties": {"artifactId": {"type": "string"}, "maxBytes": {"type": "integer", "minimum": 1, "maximum": 16777216}}, "required": ["artifactId"], "additionalProperties": False}},
    {"name": "spark_read_artifact_chunk", "description": "Read any artifact, including large GLBs, in verified base64 chunks.", "inputSchema": {"type": "object", "properties": {"artifactId": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1, "maximum": 4194304}}, "required": ["artifactId", "offset", "length"], "additionalProperties": False}},
]


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    broker = client()
    if name == "spark_capabilities":
        return broker.capabilities()
    if name == "spark_submit_job":
        return broker.submit(args["request"])
    if name == "spark_upload_artifact":
        encoded = args["base64"]
        if not isinstance(encoded, str) or len(encoded) > 90 * 1024 * 1024:
            raise ValueError("MCP artifact upload encoding exceeds the 64 MiB limit")
        data = base64.b64decode(encoded, validate=True)
        if len(data) > 64 * 1024 * 1024:
            raise ValueError("MCP artifact upload is limited to 64 MiB; use sparkctl or HTTP for larger files")
        return broker.upload_bytes(data, kind=args["kind"], role=args["role"], media_type=args["mediaType"], origin=args["origin"])
    if name == "spark_generate_3d":
        request = request_envelope(origin=args["origin"], capability="asset.3d.generate", idempotency_key=args["idempotencyKey"])
        mode = args.get("mode", "shape")
        approved = bool(args.get("authorizeBlenderContinuation", False))
        request.update({
            "priority": args.get("priority", 50),
            "inputs": [{"artifactId": args["artifactId"], "role": "source_image", "sha256": args["artifactSha256"]}],
            "requiredOutputs": [
                {"role": "shape_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ] + ([{"role": "pbr_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True}] if mode == "pbr" else []),
            "constraints": {"mode": mode, "seed": args.get("seed", 42), "maxFaces": args.get("maxFaces", 1_000_000), "targetEngine": args.get("targetEngine", "generic"), "requireWatertight": args.get("requireWatertight", False)},
            "workflow": {"autoContinue": bool(args.get("autoContinue", False)), "approvedCapabilities": ["asset.3d.prepare.blender"] if approved else [], "maxContinuations": 1 if approved else 0},
            "metadata": {"submittedByTool": "spark_generate_3d"},
        })
        return broker.submit(request)
    if name == "spark_chat":
        request = request_envelope(origin=args["origin"], capability="text.chat.generate", idempotency_key=args["idempotencyKey"])
        constraints = {"temperature": args.get("temperature", 0.2), "maxTokens": args.get("maxTokens", 1024), "systemPrompt": args.get("systemPrompt", ""), "timeoutSeconds": args.get("timeoutSeconds", 600), "enableThinking": args.get("enableThinking", False)}
        for key in ("model", "serviceClass", "routePreference"):
            if key in args:
                constraints[key] = args[key]
        request.update({
            "priority": args.get("priority", 50),
            "inputs": [{"artifactId": args["artifactId"], "role": "prompt", "sha256": args["artifactSha256"]}],
            "requiredOutputs": [
                {"role": "text_output", "kind": "text", "mediaTypes": ["text/plain"], "required": True},
                {"role": "provider_response", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ],
            "constraints": constraints,
            "metadata": {"submittedByTool": "spark_chat"},
        })
        return broker.submit(request)
    if name == "spark_job_status":
        return broker.job(args["jobId"])
    if name == "spark_job_events":
        return broker.events(args["jobId"])
    if name == "spark_cancel_job":
        return broker.cancel(args["jobId"])
    if name == "spark_artifact_metadata":
        return broker.artifact(args["artifactId"])
    if name == "spark_download_artifact":
        metadata = broker.artifact(args["artifactId"])["artifact"]
        maximum = args.get("maxBytes", 16 * 1024 * 1024)
        if metadata["sizeBytes"] > maximum:
            raise ValueError(f"artifact is {metadata['sizeBytes']} bytes, above maxBytes={maximum}")
        data = broker.download_bytes(args["artifactId"])
        return {"artifact": metadata, "base64": base64.b64encode(data).decode("ascii")}
    if name == "spark_read_artifact_chunk":
        metadata = broker.artifact(args["artifactId"])["artifact"]
        data, headers = broker.download_chunk(args["artifactId"], offset=args["offset"], length=args["length"])
        return {
            "artifactId": args["artifactId"], "artifactSha256": metadata["sha256"], "sizeBytes": metadata["sizeBytes"],
            "offset": args["offset"], "length": len(data), "chunkSha256": hashlib.sha256(data).hexdigest(),
            "base64": base64.b64encode(data).decode("ascii"), "eof": args["offset"] + len(data) >= metadata["sizeBytes"],
        }
    raise ValueError(f"unknown tool: {name}")


def response(request_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> None:
    value: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        value["error"] = error
    else:
        value["result"] = result
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return
    if method == "initialize":
        response(request_id, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {"listChanged": False}}, "serverInfo": {"name": "go7-spark-broker", "version": BROKER_VERSION}})
        return
    if method == "ping":
        response(request_id, {})
        return
    if method == "tools/list":
        response(request_id, {"tools": TOOLS})
        return
    if method == "tools/call":
        params = message.get("params", {})
        try:
            result = call_tool(params.get("name", ""), params.get("arguments", {}))
            response(request_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2, sort_keys=True)}], "structuredContent": result, "isError": False})
        except (ClientError, ValueError, KeyError, RuntimeError) as exc:
            code = exc.code if isinstance(exc, ClientError) else "tool_error"
            response(request_id, {"content": [{"type": "text", "text": json.dumps({"error": {"code": code, "message": str(exc)}})}], "isError": True})
        return
    response(request_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> None:
    maximum_line = 96 * 1024 * 1024
    while True:
        raw = sys.stdin.buffer.readline(maximum_line + 1)
        if not raw:
            break
        if len(raw) > maximum_line:
            while raw and not raw.endswith(b"\n"):
                raw = sys.stdin.buffer.readline(maximum_line + 1)
            continue
        try:
            line = raw.decode("utf-8")
            message = json.loads(line)
            if isinstance(message, dict):
                handle(message)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


if __name__ == "__main__":
    main()
