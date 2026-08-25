#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from spark_broker.mcp_server import call_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Live MCP model3d upload and chunked download verifier")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--origin", default="spark-mcp-live-test")
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    uploaded = call_tool("spark_upload_artifact", {
        "base64": base64.b64encode(source_bytes).decode("ascii"),
        "kind": "model3d",
        "role": "roundtrip_model",
        "mediaType": "model/gltf-binary",
        "origin": args.origin,
    })["artifact"]
    if uploaded["sha256"] != source_sha:
        raise SystemExit("MCP upload hash mismatch")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = args.destination.with_name(f".{args.destination.name}.partial")
    digest = hashlib.sha256()
    received = 0
    chunks = 0
    with partial.open("wb") as target:
        while received < uploaded["sizeBytes"]:
            result = call_tool("spark_read_artifact_chunk", {
                "artifactId": uploaded["id"], "offset": received, "length": 4 * 1024 * 1024,
            })
            data = base64.b64decode(result["base64"], validate=True)
            if hashlib.sha256(data).hexdigest() != result["chunkSha256"]:
                raise SystemExit("MCP chunk hash mismatch")
            target.write(data)
            digest.update(data)
            received += len(data)
            chunks += 1
        target.flush()
    if digest.hexdigest() != uploaded["sha256"] or received != uploaded["sizeBytes"]:
        partial.unlink(missing_ok=True)
        raise SystemExit("MCP reconstructed artifact failed integrity check")
    partial.replace(args.destination)
    print(json.dumps({
        "artifactId": uploaded["id"], "mediaType": uploaded["mediaType"], "role": uploaded["role"],
        "sizeBytes": received, "sha256": digest.hexdigest(), "chunks": chunks, "roundtripVerified": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
