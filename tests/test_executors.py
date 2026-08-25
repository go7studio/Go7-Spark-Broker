from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from spark_broker.artifacts import ArtifactRegistry
from spark_broker.contract import ContractError, validate_job_request
from spark_broker.executors import ExecutionFailure, Hunyuan3DExecutor, OpenAIChatExecutor
from spark_broker.store import Store
from tests.helpers import request


class FakeTextHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.headers.get("Authorization") != "Bearer test-key":
            self.send_response(401)
        elif self.path == "/readyz":
            self.send_response(200)
        else:
            self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        prompt = body["messages"][-1]["content"]
        response = json.dumps({
            "id": "chatcmpl-test", "object": "chat.completion", "model": "local-test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"MODEL_OK:{prompt}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


class FakeComposeRunner:
    def __init__(self, root: Path, *, faces: int = 40000, watertight_geometry: int = 1) -> None:
        self.root = root
        self.faces = faces
        self.watertight_geometry = watertight_geometry
        self.commands: list[list[str]] = []

    def run(self, argv: list[str], **kwargs: Any) -> int:
        self.commands.append(argv)
        if "scripts/shape_infer.py" in argv:
            output = argv[argv.index("--out") + 1].removeprefix("/workspace/out/")
            (self.root / "out" / output).write_bytes(b"glTFfake-shape")
        elif "scripts/texture_infer.py" in argv:
            output = argv[argv.index("--out") + 1].removeprefix("/workspace/out/")
            (self.root / "out" / output).write_bytes(b"glTFfake-pbr")
        elif "scripts/validate_mesh.py" in argv:
            output = argv[argv.index("--json") + 1].removeprefix("/workspace/out/")
            (self.root / "out" / output).write_text(json.dumps({
                "valid": True, "faces": self.faces, "vertices": 20002, "geometry_count": 1,
                "watertight_geometry": self.watertight_geometry, "bounds": [[0, 0, 0], [1, 1, 1]], "materials": ["PBRMaterial"],
            }))
        return 0


class ExecutorContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "db.sqlite3")
        self.registry = ArtifactRegistry(self.root / "registry", self.store, max_upload_bytes=64 * 1024 * 1024)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_capabilities_publish_complete_generic_invocation_contracts(self) -> None:
        text = OpenAIChatExecutor(
            endpoint="http://127.0.0.1:1", api_key="test-key", model="local-test-model",
            container_name=None, profile_id="gpu.test-profile", estimated_memory_gb=48,
        ).capability.public()
        self.assertEqual(text["profileId"], "gpu.test-profile")
        self.assertEqual(text["estimatedMemoryGb"], 48)
        self.assertEqual(text["invocation"]["inputs"][0], {
            "role": "prompt", "kind": "text", "mediaTypes": ["text/plain", "application/json"],
            "required": True, "minItems": 1, "maxItems": 1,
        })
        self.assertEqual([item["role"] for item in text["invocation"]["outputs"] if item["required"]], ["text_output", "provider_response"])
        self.assertFalse(text["invocation"]["constraintsSchema"]["additionalProperties"])

        model = Hunyuan3DExecutor.capability.public()
        self.assertEqual(model["invocation"]["inputs"][0]["role"], "source_image")
        self.assertEqual([item["role"] for item in model["invocation"]["outputs"] if item["required"]], ["shape_model", "mesh_report"])
        self.assertEqual(model["continuations"][0]["capability"], "asset.3d.prepare.blender")
        self.assertEqual(model["continuations"][0]["outputs"][0]["mediaTypes"], ["model/gltf-binary"])

    def upload(self, data: bytes, *, kind: str, media: str) -> dict[str, Any]:
        return self.registry.import_stream(io.BytesIO(data), size=len(data), kind=kind, role="input", media_type=media)

    def test_openai_executor_returns_typed_verified_text_artifacts(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTextHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            prompt = self.upload(b"router test", kind="text", media="text/plain")
            value = validate_job_request(request(
                capability="text.chat.generate",
                inputs=[{"artifactId": prompt["id"], "sha256": prompt["sha256"], "role": "prompt"}],
                requiredOutputs=[
                    {"role": "text_output", "kind": "text", "mediaTypes": ["text/plain"], "required": True},
                    {"role": "provider_response", "kind": "report", "mediaTypes": ["application/json"], "required": True},
                ],
                constraints={"temperature": 0, "maxTokens": 64, "timeoutSeconds": 30},
            ), broker_id="spark.test")
            job, _ = self.store.submit(value)
            executor = OpenAIChatExecutor(endpoint=f"http://127.0.0.1:{server.server_address[1]}", api_key="test-key", model="local-test-model", container_name=None)
            executor.activate(lambda: False)
            result = executor.execute(job, self.registry, lambda: False, lambda _state, _detail: None)
            self.assertEqual([item["role"] for item in result["artifacts"]], ["text_output", "provider_response"])
            text_meta, text_path = self.registry.resolve(result["data"]["primaryArtifactId"], verify=True)
            self.assertEqual(text_meta["validation"]["usage"]["total_tokens"], 7)
            self.assertEqual(text_path.read_text(), "MODEL_OK:router test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_openai_executor_rejects_incompatible_required_output(self) -> None:
        prompt = self.upload(b"router test", kind="text", media="text/plain")
        value = validate_job_request(request(
            capability="text.chat.generate",
            inputs=[{"artifactId": prompt["id"], "sha256": prompt["sha256"], "role": "prompt"}],
            requiredOutputs=[
                {"role": "text_output", "kind": "text", "mediaTypes": ["application/json"], "required": True},
            ],
        ), broker_id="spark.test")
        job, _ = self.store.submit(value)
        executor = OpenAIChatExecutor(endpoint="http://127.0.0.1:1", api_key="test-key", model="local-test-model", container_name=None)
        with self.assertRaisesRegex(ContractError, "supported output definition") as context:
            executor.execute(job, self.registry, lambda: False, lambda _state, _detail: None)
        self.assertEqual(context.exception.code, "unsupported_media_type")

    def test_openai_activation_does_not_restart_an_already_running_container(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTextHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            executor = OpenAIChatExecutor(
                endpoint=f"http://127.0.0.1:{server.server_address[1]}", api_key="test-key",
                model="local-test-model", container_name="local-text-runtime",
            )
            with patch("spark_broker.executors.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, stdout="true\n")
                executor.activate(lambda: False)
            self.assertEqual(run.call_args_list[0].args[0], ["docker", "inspect", "--format", "{{.State.Running}}", "local-text-runtime"])
            self.assertEqual(len(run.call_args_list), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def test_3d_executor_returns_models_reports_and_authorized_continuation(self) -> None:
        workload = self.root / "hunyuan"
        (workload / "assets").mkdir(parents=True)
        (workload / "out").mkdir()
        (workload / "compose.yaml").write_text("services: {}\n")
        source = self.upload(b"fake png", kind="image", media="image/png")
        value = validate_job_request(request(
            capability="asset.3d.generate",
            inputs=[{"artifactId": source["id"], "sha256": source["sha256"], "role": "source_image"}],
            requiredOutputs=[
                {"role": "shape_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "pbr_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ],
            constraints={"mode": "pbr", "maxFaces": 100000, "targetEngine": "godot"},
            workflow={"autoContinue": True, "approvedCapabilities": ["asset.3d.prepare.blender"], "maxContinuations": 1},
        ), broker_id="spark.test")
        job, _ = self.store.submit(value)
        executor = Hunyuan3DExecutor(workload, broker_id="spark.test", stop_containers=())
        fake = FakeComposeRunner(workload)
        executor.runner = fake
        result = executor.execute(job, self.registry, lambda: False, lambda _state, _detail: None)
        roles = {item["role"] for item in result["artifacts"]}
        self.assertTrue({"shape_model", "pbr_model", "mesh_report", "pbr_mesh_report"}.issubset(roles))
        continuation = result["continuations"][0]
        self.assertEqual(continuation["capability"], "asset.3d.prepare.blender")
        self.assertTrue(continuation["authorization"]["approvedByRequest"])
        self.assertTrue(continuation["autoStartEligible"])
        self.assertEqual({item["role"] for item in continuation["requiredOutputs"]}, {"game_model", "preview", "blender_report"})
        primary, path = self.registry.resolve(result["data"]["primaryArtifactId"], verify=True)
        self.assertEqual(primary["mediaType"], "model/gltf-binary")
        self.assertEqual(path.read_bytes(), b"glTFfake-pbr")
        flat_commands = " ".join(" ".join(command) for command in fake.commands)
        self.assertIn("shape_infer.py", flat_commands)
        self.assertIn("texture_infer.py", flat_commands)

    def test_3d_executor_returns_high_poly_intermediate_only_for_authorized_blender_repair(self) -> None:
        workload = self.root / "hunyuan-high-poly"
        (workload / "assets").mkdir(parents=True)
        (workload / "out").mkdir()
        (workload / "compose.yaml").write_text("services: {}\n")
        source = self.upload(b"fake png", kind="image", media="image/png")

        def submitted(approved: bool) -> dict[str, Any]:
            value = validate_job_request(request(
                capability="asset.3d.generate",
                inputs=[{"artifactId": source["id"], "sha256": source["sha256"], "role": "source_image"}],
                requiredOutputs=[
                    {"role": "shape_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                    {"role": "mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
                ],
                constraints={"mode": "shape", "maxFaces": 100000, "targetEngine": "godot", "requireWatertight": True},
                workflow={
                    "autoContinue": False,
                    "approvedCapabilities": ["asset.3d.prepare.blender"] if approved else [],
                    "maxContinuations": 1 if approved else 0,
                },
                idempotencyKey=f"high-poly-{approved}",
            ), broker_id="spark.test")
            return self.store.submit(value)[0]

        approved_executor = Hunyuan3DExecutor(workload, broker_id="spark.test", stop_containers=())
        approved_executor.runner = FakeComposeRunner(workload, faces=740092, watertight_geometry=0)
        result = approved_executor.execute(submitted(True), self.registry, lambda: False, lambda _state, _detail: None)
        primary, _path = self.registry.resolve(result["data"]["primaryArtifactId"], verify=True)
        self.assertFalse(primary["validation"]["faceLimitSatisfied"])
        self.assertFalse(primary["validation"]["watertightSatisfied"])
        self.assertTrue(primary["validation"]["requiresPreparation"])
        self.assertTrue(result["data"]["requiresPreparation"])
        self.assertEqual(result["continuations"][0]["constraints"]["targetFaces"], 100000)
        report_artifact = next(artifact for artifact in result["artifacts"] if artifact["role"] == "mesh_report")
        _report_metadata, report_path = self.registry.resolve(report_artifact["id"], verify=True)
        persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(persisted_report["faceLimitSatisfied"])
        self.assertFalse(persisted_report["watertightSatisfied"])
        self.assertTrue(persisted_report["requiresPreparation"])

        denied_executor = Hunyuan3DExecutor(workload, broker_id="spark.test", stop_containers=())
        denied_executor.runner = FakeComposeRunner(workload, faces=740092)
        with self.assertRaisesRegex(ExecutionFailure, "exceeds 100000 faces") as context:
            denied_executor.execute(submitted(False), self.registry, lambda: False, lambda _state, _detail: None)
        self.assertEqual(context.exception.code, "face_limit_exceeded")

    def test_3d_activation_removes_only_containers_owned_by_this_broker(self) -> None:
        workload = self.root / "hunyuan-activate"
        workload.mkdir()
        executor = Hunyuan3DExecutor(workload, broker_id="spark.test")
        with patch("spark_broker.executors.subprocess.run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout="a" * 12 + "\nnot-a-container\n"),
                subprocess.CompletedProcess([], 0),
            ]
            executor.activate(lambda: False)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ["docker", "ps", "-aq", "--filter", "label=go7.spark-broker.owner=spark.test"])
        self.assertEqual(commands[1], ["docker", "rm", "-f", "a" * 12])

    def test_3d_executor_rejects_cross_profile_container_stop_configuration(self) -> None:
        workload = self.root / "hunyuan-unsafe-stop"
        workload.mkdir()
        with self.assertRaisesRegex(ValueError, "fenced resource controllers"):
            Hunyuan3DExecutor(workload, broker_id="spark.test", stop_containers=("local-text-runtime",))

    def test_3d_activation_surfaces_docker_timeouts_as_retryable_failures(self) -> None:
        workload = self.root / "hunyuan-timeout"
        workload.mkdir()
        executor = Hunyuan3DExecutor(workload, broker_id="spark.test")
        with patch("spark_broker.executors.subprocess.run", side_effect=subprocess.TimeoutExpired(["docker", "ps"], 30)):
            with self.assertRaises(ExecutionFailure) as context:
                executor.activate(lambda: False)
        self.assertEqual(context.exception.code, "docker_timeout")
        self.assertTrue(context.exception.retryable)


if __name__ == "__main__":
    unittest.main()
