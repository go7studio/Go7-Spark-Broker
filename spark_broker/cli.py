from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .client import BrokerClient, ClientError, read_credential
from .routing import RoutingError, compile_routing_config, simulate_routing_scenarios


TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def client_from_args(args: argparse.Namespace) -> BrokerClient:
    token = args.token or os.environ.get("SPARK_BROKER_TOKEN", "")
    if not token and args.token_file:
        token = read_credential(Path(args.token_file))
    if not token:
        raise SystemExit("set SPARK_BROKER_TOKEN or pass --token-file")
    return BrokerClient(args.url, token, timeout=args.timeout)


def request_envelope(*, origin: str, capability: str, idempotency_key: str | None = None) -> dict[str, Any]:
    unique = uuid.uuid4().hex
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "requestId": f"req_{unique}",
        "traceId": f"trace_{unique}",
        "idempotencyKey": idempotency_key or f"idem_{unique}",
        "origin": origin,
        "visitedSystems": [],
        "hopCount": 0,
        "capability": capability,
        "priority": 50,
        "inputs": [],
        "requiredOutputs": [],
        "constraints": {},
        "workflow": {"autoContinue": False, "approvedCapabilities": [], "maxContinuations": 0},
        "metadata": {},
    }


def wait_for_job(client: BrokerClient, job_id: str, interval: float, timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    last = ""
    while True:
        job = client.job(job_id)
        if job["status"] != last:
            print(f"{job_id}: {job['status']}", file=sys.stderr)
            last = job["status"]
        if job["status"] in TERMINAL:
            return job
        if time.monotonic() - started > timeout:
            raise SystemExit(f"timed out waiting for {job_id}")
        time.sleep(interval)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sparkctl", description="Typed client for the Go7 Spark capability broker")
    root.add_argument("--url", default=os.environ.get("SPARK_BROKER_URL", "http://127.0.0.1:8790"))
    root.add_argument("--token", default="", help=argparse.SUPPRESS)
    root.add_argument("--token-file", default=os.environ.get("SPARK_BROKER_TOKEN_FILE", ""))
    root.add_argument("--timeout", type=int, default=60)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    sub.add_parser("status")

    route_validate = sub.add_parser("route-validate", help="validate and fingerprint a routing config without contacting the broker")
    route_validate.add_argument("config", type=Path)
    route_simulate = sub.add_parser("route-simulate", help="run a pure routing scenario without contacting runtimes")
    route_simulate.add_argument("scenario", type=Path)

    upload = sub.add_parser("upload")
    upload.add_argument("path", type=Path)
    upload.add_argument("--kind", required=True)
    upload.add_argument("--role", required=True)
    upload.add_argument("--media-type", default="")
    upload.add_argument("--origin", default="sparkctl")

    submit = sub.add_parser("submit")
    submit.add_argument("request", type=Path, help="protocol v1 request JSON")

    generate = sub.add_parser("generate-3d")
    generate.add_argument("image", type=Path)
    generate.add_argument("--mode", choices=["shape", "pbr"], default="shape")
    generate.add_argument("--origin", default="sparkctl")
    generate.add_argument("--idempotency-key", default="")
    generate.add_argument("--priority", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--max-faces", type=int, default=1_000_000)
    generate.add_argument("--target-engine", choices=["generic", "blender", "godot", "unity", "unreal"], default="godot")
    generate.add_argument("--require-watertight", action="store_true")
    generate.add_argument("--approve-blender", action="store_true")
    generate.add_argument("--auto-continue", action="store_true")
    generate.add_argument("--wait", action="store_true")
    generate.add_argument("--wait-timeout", type=int, default=3600)

    chat = sub.add_parser("chat")
    chat.add_argument("prompt", nargs="?", default="")
    chat.add_argument("--prompt-file", type=Path)
    chat.add_argument("--origin", default="sparkctl")
    chat.add_argument("--idempotency-key", default="")
    chat.add_argument("--priority", type=int, default=50)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--max-tokens", type=int, default=1024)
    chat.add_argument("--system-prompt", default="")
    chat.add_argument("--enable-thinking", action="store_true")
    chat.add_argument("--model", default="", help="advertised local model identity; never an endpoint or container")
    chat.add_argument("--service-class", choices=["interactive", "batch", "background"], default="interactive")
    chat.add_argument("--route-preference", choices=["balanced", "latency", "throughput", "memory"], default="balanced")
    chat.add_argument("--wait", action="store_true")
    chat.add_argument("--print-output", action="store_true")
    chat.add_argument("--wait-timeout", type=int, default=1800)

    job = sub.add_parser("job")
    job.add_argument("job_id")
    events = sub.add_parser("events")
    events.add_argument("job_id")
    wait = sub.add_parser("wait")
    wait.add_argument("job_id")
    wait.add_argument("--interval", type=float, default=2)
    wait.add_argument("--wait-timeout", type=int, default=3600)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    artifact = sub.add_parser("artifact")
    artifact.add_argument("artifact_id")
    download = sub.add_parser("download")
    download.add_argument("artifact_id")
    download.add_argument("destination", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command in {"route-validate", "route-simulate"}:
        path = args.config if args.command == "route-validate" else args.scenario
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            result = (
                compile_routing_config(value).public()
                if args.command == "route-validate"
                else simulate_routing_scenarios(value)
            )
        except (OSError, json.JSONDecodeError, RoutingError) as exc:
            print(json.dumps({"error": {"code": "invalid_routing_input", "message": str(exc)}}), file=sys.stderr)
            raise SystemExit(2) from exc
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    client = client_from_args(args)
    try:
        if args.command == "capabilities":
            result = client.capabilities()
        elif args.command == "status":
            result = client.status()
        elif args.command == "upload":
            media = args.media_type or mimetypes.guess_type(args.path.name)[0] or "application/octet-stream"
            result = client.upload(args.path, kind=args.kind, role=args.role, media_type=media, origin=args.origin)
        elif args.command == "submit":
            result = client.submit(json.loads(args.request.read_text(encoding="utf-8")))
        elif args.command == "generate-3d":
            media = mimetypes.guess_type(args.image.name)[0] or "application/octet-stream"
            uploaded = client.upload(args.image, kind="image", role="source_image", media_type=media, origin=args.origin)
            artifact = uploaded["artifact"]
            request = request_envelope(origin=args.origin, capability="asset.3d.generate", idempotency_key=args.idempotency_key or None)
            request.update({
                "priority": args.priority,
                "inputs": [{"artifactId": artifact["id"], "role": "source_image", "sha256": artifact["sha256"]}],
                "requiredOutputs": [
                    {"role": "shape_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                    {"role": "mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
                ] + ([{"role": "pbr_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True}] if args.mode == "pbr" else []),
                "constraints": {"mode": args.mode, "seed": args.seed, "maxFaces": args.max_faces, "targetEngine": args.target_engine, "requireWatertight": args.require_watertight},
                "workflow": {"autoContinue": args.auto_continue, "approvedCapabilities": ["asset.3d.prepare.blender"] if args.approve_blender else [], "maxContinuations": 1 if args.approve_blender else 0},
                "metadata": {"sourceName": args.image.name},
            })
            result = client.submit(request)
            if args.wait:
                result = wait_for_job(client, result["id"], 2, args.wait_timeout)
        elif args.command == "chat":
            if args.prompt_file:
                prompt_bytes = args.prompt_file.read_bytes()
            elif args.prompt:
                prompt_bytes = args.prompt.encode("utf-8")
            else:
                raise SystemExit("provide a prompt or --prompt-file")
            uploaded = client.upload_bytes(prompt_bytes, kind="text", role="prompt", media_type="text/plain", origin=args.origin)
            artifact = uploaded["artifact"]
            request = request_envelope(origin=args.origin, capability="text.chat.generate", idempotency_key=args.idempotency_key or None)
            constraints = {
                "temperature": args.temperature,
                "maxTokens": args.max_tokens,
                "systemPrompt": args.system_prompt,
                "timeoutSeconds": min(args.wait_timeout, 1800),
                "enableThinking": args.enable_thinking,
            }
            if args.model:
                constraints["model"] = args.model
            if args.service_class != "interactive":
                constraints["serviceClass"] = args.service_class
            if args.route_preference != "balanced":
                constraints["routePreference"] = args.route_preference
            request.update({
                "priority": args.priority,
                "inputs": [{"artifactId": artifact["id"], "role": "prompt", "sha256": artifact["sha256"]}],
                "requiredOutputs": [
                    {"role": "text_output", "kind": "text", "mediaTypes": ["text/plain"], "required": True},
                    {"role": "provider_response", "kind": "report", "mediaTypes": ["application/json"], "required": True},
                ],
                "constraints": constraints,
                "metadata": {"submittedBy": "sparkctl.chat"},
            })
            result = client.submit(request)
            if args.wait or args.print_output:
                result = wait_for_job(client, result["id"], 2, args.wait_timeout)
            if args.print_output and result["status"] == "completed":
                primary = result["result"]["data"]["primaryArtifactId"]
                sys.stdout.write(client.download_bytes(primary).decode("utf-8") + "\n")
                return
        elif args.command == "job":
            result = client.job(args.job_id)
        elif args.command == "events":
            result = client.events(args.job_id)
        elif args.command == "wait":
            result = wait_for_job(client, args.job_id, args.interval, args.wait_timeout)
        elif args.command == "cancel":
            result = client.cancel(args.job_id)
        elif args.command == "artifact":
            result = client.artifact(args.artifact_id)
        elif args.command == "download":
            client.download(args.artifact_id, args.destination)
            result = {"artifactId": args.artifact_id, "destination": str(args.destination), "downloaded": True}
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, indent=2, sort_keys=True))
    except ClientError as exc:
        print(json.dumps({"error": {"status": exc.status, "code": exc.code, "message": str(exc)}}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
