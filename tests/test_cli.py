from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from spark_broker.server import Broker, BrokerHTTPServer, Config
from tests.helpers import HealthyHostProbe, write_resource_policy
from tests.test_executors import FakeTextHandler


class CLIFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeTextHandler)
        self.provider_thread = threading.Thread(target=self.provider.serve_forever, daemon=True)
        self.provider_thread.start()
        provider_url = f"http://127.0.0.1:{self.provider.server_address[1]}"
        policy = write_resource_policy(self.root, provider_url)
        config = Config(
            broker_id="spark.cli-test", bind="127.0.0.1", port=0, token="c" * 32,
            data_root=self.root / "data", hunyuan_root=None, stop_containers=(),
            text_endpoint=provider_url, text_api_key="test-key",
            text_model="local-test-model", text_container=None, max_artifact_bytes=64 * 1024 * 1024,
            resource_policy_file=policy, coordinator_lock_file=self.root / "gpu0.lock",
            coordinator_epoch_file=self.root / "gpu0.epoch",
        )
        self.broker = Broker(config)
        self.broker.coordinator.host_probe = HealthyHostProbe()
        self.http = BrokerHTTPServer((config.bind, 0), self.broker)
        self.broker.start()
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        self.base = [
            sys.executable, "-m", "spark_broker.cli", "--url", f"http://127.0.0.1:{self.http.server_address[1]}",
            "--token", config.token, "--timeout", "30",
        ]

    def tearDown(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.broker.stop()
        self.http_thread.join(5)
        self.provider.shutdown()
        self.provider.server_close()
        self.provider_thread.join(5)
        self.temporary.cleanup()

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
        return subprocess.run([*self.base, *args], cwd=Path(__file__).parents[1], env=environment, text=True, capture_output=True, timeout=60, check=True)

    def test_cli_model_call_prints_verified_model_output(self) -> None:
        result = self.run_cli("chat", "cli router test", "--temperature", "0", "--max-tokens", "32", "--wait", "--print-output")
        self.assertEqual(result.stdout.strip(), "MODEL_OK:cli router test")
        self.assertIn("completed", result.stderr)

    def test_cli_sends_and_receives_glb_unchanged(self) -> None:
        source = self.root / "source.glb"
        destination = self.root / "received.glb"
        payload = b"glTF" + bytes(range(251)) * 4000
        source.write_bytes(payload)
        uploaded = json.loads(self.run_cli("upload", str(source), "--kind", "model3d", "--role", "source_model", "--media-type", "model/gltf-binary").stdout)
        artifact = uploaded["artifact"]
        downloaded = json.loads(self.run_cli("download", artifact["id"], str(destination)).stdout)
        self.assertTrue(downloaded["downloaded"])
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), artifact["sha256"])
        self.assertEqual(list(self.root.glob(".received.glb.*.partial")), [])


if __name__ == "__main__":
    unittest.main()
