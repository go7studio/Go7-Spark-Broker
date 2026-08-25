from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from spark_broker.mcp_server import call_tool
from spark_broker.server import Broker, BrokerHTTPServer, Config
from tests.test_executors import FakeTextHandler


class MCPFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeTextHandler)
        self.provider_thread = threading.Thread(target=self.provider.serve_forever, daemon=True)
        self.provider_thread.start()
        config = Config(
            broker_id="spark.mcp-test", bind="127.0.0.1", port=0, token="m" * 32,
            data_root=Path(self.temporary.name), hunyuan_root=None, stop_containers=(),
            text_endpoint=f"http://127.0.0.1:{self.provider.server_address[1]}", text_api_key="test-key",
            text_model="local-test-model", text_container=None, max_artifact_bytes=64 * 1024 * 1024,
        )
        self.broker = Broker(config)
        self.http = BrokerHTTPServer((config.bind, 0), self.broker)
        self.broker.start()
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        self.previous_url = os.environ.get("SPARK_BROKER_URL")
        self.previous_token = os.environ.get("SPARK_BROKER_TOKEN")
        os.environ["SPARK_BROKER_URL"] = f"http://127.0.0.1:{self.http.server_address[1]}"
        os.environ["SPARK_BROKER_TOKEN"] = config.token

    def tearDown(self) -> None:
        if self.previous_url is None:
            os.environ.pop("SPARK_BROKER_URL", None)
        else:
            os.environ["SPARK_BROKER_URL"] = self.previous_url
        if self.previous_token is None:
            os.environ.pop("SPARK_BROKER_TOKEN", None)
        else:
            os.environ["SPARK_BROKER_TOKEN"] = self.previous_token
        self.http.shutdown()
        self.http.server_close()
        self.broker.stop()
        self.http_thread.join(5)
        self.provider.shutdown()
        self.provider.server_close()
        self.provider_thread.join(5)
        self.temporary.cleanup()

    def wait(self, job_id: str) -> dict:
        for _ in range(200):
            job = call_tool("spark_job_status", {"jobId": job_id})
            if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return job
            time.sleep(0.01)
        self.fail("MCP job did not finish")

    def test_mcp_upload_model_call_and_chunked_text_receive(self) -> None:
        uploaded = call_tool("spark_upload_artifact", {
            "base64": base64.b64encode(b"mcp router test").decode(), "kind": "text", "role": "prompt",
            "mediaType": "text/plain", "origin": "mcp-tests",
        })["artifact"]
        submitted = call_tool("spark_chat", {
            "artifactId": uploaded["id"], "artifactSha256": uploaded["sha256"], "origin": "mcp-tests",
            "idempotencyKey": "mcp-model-test", "temperature": 0, "maxTokens": 32,
        })
        finished = self.wait(submitted["id"])
        self.assertEqual(finished["status"], "completed")
        primary = finished["result"]["data"]["primaryArtifactId"]
        chunk = call_tool("spark_read_artifact_chunk", {"artifactId": primary, "offset": 0, "length": 4096})
        self.assertTrue(chunk["eof"])
        self.assertEqual(base64.b64decode(chunk["base64"]), b"MODEL_OK:mcp router test")
        self.assertEqual(chunk["artifactSha256"], finished["result"]["artifacts"][0]["sha256"])

    def test_mcp_can_send_and_receive_glb_bytes_with_hashes(self) -> None:
        glb = b"glTF" + bytes(range(256)) * 20000
        uploaded = call_tool("spark_upload_artifact", {
            "base64": base64.b64encode(glb).decode(), "kind": "model3d", "role": "source_model",
            "mediaType": "model/gltf-binary", "origin": "mcp-tests",
        })["artifact"]
        received = bytearray()
        offset = 0
        while offset < uploaded["sizeBytes"]:
            chunk = call_tool("spark_read_artifact_chunk", {"artifactId": uploaded["id"], "offset": offset, "length": 1024 * 1024})
            decoded = base64.b64decode(chunk["base64"])
            received.extend(decoded)
            offset += len(decoded)
        self.assertEqual(bytes(received), glb)
        self.assertEqual(chunk["artifactSha256"], uploaded["sha256"])
        self.assertTrue(chunk["eof"])


if __name__ == "__main__":
    unittest.main()
