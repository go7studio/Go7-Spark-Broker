from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from spark_broker.artifacts import ArtifactRegistry
from spark_broker.contract import ContractError, validate_job_request
from spark_broker.executors import OpenAIRoute, RoutedOpenAIChatExecutor, load_openai_routes
from spark_broker.routing import compile_routing_config
from spark_broker.store import Store
from tests.helpers import request


class InferenceRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "db.sqlite3")
        self.registry = ArtifactRegistry(self.root / "artifacts", self.store, max_upload_bytes=1024 * 1024)
        self.prompt = self.registry.import_stream(
            io.BytesIO(b"route me"), size=8, kind="text", role="prompt", media_type="text/plain"
        )
        self.routes = (
            OpenAIRoute(
                id="fast-small", model="small-model", profile_id="gpu.small", description="Small low-latency model",
                endpoint="http://127.0.0.1:8101", api_key="test-key", container_name=None,
                estimated_memory_gb=20, priority=90, service_classes=("interactive", "batch"),
            ),
            OpenAIRoute(
                id="large-quality", model="large-model", profile_id="gpu.large", description="Large quality model",
                endpoint="http://127.0.0.1:8102", api_key="test-key", container_name=None,
                estimated_memory_gb=72, priority=70, service_classes=("interactive", "batch", "background"),
            ),
        )
        self.executor = RoutedOpenAIChatExecutor(self.routes)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def job(self, constraints=None):
        value = validate_job_request(request(
            capability="text.chat.generate",
            inputs=[{"artifactId": self.prompt["id"], "sha256": self.prompt["sha256"], "role": "prompt"}],
            requiredOutputs=[
                {"role": "text_output", "kind": "text", "mediaTypes": ["text/plain"], "required": True},
                {"role": "provider_response", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ],
            constraints=constraints or {},
        ), broker_id="spark.test")
        return self.store.submit(value)[0]

    def test_default_route_uses_admin_priority_and_persists_reason(self) -> None:
        job = self.job()
        self.executor.validate_request(job, self.registry)
        plan = self.executor.plan(job)
        self.assertEqual(plan.route_id, "fast-small")
        self.assertEqual(plan.profile_id, "gpu.small")
        self.assertEqual(plan.route_reason, "policy:balanced")

    def test_balanced_policy_prefers_a_healthy_resident_profile(self) -> None:
        plan = self.executor.plan(self.job(), {
            "activeProfiles": ["gpu.large"],
            "profiles": {"gpu.large": {"health": "healthy"}, "gpu.small": {"health": "healthy"}},
        })
        self.assertEqual(plan.route_id, "large-quality")

    def test_explicit_advertised_model_selects_profile_not_endpoint(self) -> None:
        plan = self.executor.plan(self.job({"model": "large-model", "serviceClass": "background"}))
        self.assertEqual(plan.route_id, "large-quality")
        self.assertEqual(plan.service_class, "background")
        self.assertEqual(plan.route_reason, "explicit_model")
        with self.assertRaises(ContractError) as context:
            self.executor.plan(self.job({"model": "http://127.0.0.1:9999"}))
        self.assertEqual(context.exception.code, "route_unavailable")

    def test_capability_discovery_exposes_safe_route_metadata_only(self) -> None:
        routes = self.executor.public_routes()
        serialized = json.dumps(routes)
        self.assertIn("small-model", serialized)
        self.assertNotIn("127.0.0.1", serialized)
        self.assertNotIn("test-key", serialized)
        self.assertEqual(self.executor.capability.public()["resourcePolicy"]["serviceClass"], "interactive")

    def test_routes_file_is_strict_and_credential_file_backed(self) -> None:
        key = self.root / "route-key"
        key.write_text("test-key", encoding="utf-8")
        key.chmod(0o600)
        route_file = self.root / "routes.json"
        route_file.write_text(json.dumps({
            "version": 1,
            "routes": [{
                "id": "local-small", "model": "model-a", "profileId": "gpu.model-a",
                "description": "Local test route", "endpoint": "http://127.0.0.1:8000",
                "apiKeyFile": str(key), "estimatedMemoryGb": 24, "priority": 50,
                "serviceClasses": ["interactive"],
            }],
        }), encoding="utf-8")
        route_file.chmod(0o600)
        compiled = compile_routing_config(json.loads(route_file.read_text(encoding="utf-8")))
        loaded = load_openai_routes(route_file)
        self.assertEqual(loaded[0].id, "local-small")
        self.assertEqual(loaded[0].api_key, "test-key")
        executor = RoutedOpenAIChatExecutor(loaded)
        self.assertEqual(executor.installed_config_revision, compiled.revision)
        self.assertEqual(executor.public_routes()[0]["configRevision"], compiled.revision)
        self.assertNotEqual(executor.routing_config.revision, compiled.revision)
        unsafe = json.loads(route_file.read_text(encoding="utf-8"))
        unsafe["routes"][0]["endpoint"] = "https://public.example.invalid"
        route_file.write_text(json.dumps(unsafe), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "loopback"):
            load_openai_routes(route_file)


if __name__ == "__main__":
    unittest.main()
