from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from spark_broker.client import BrokerClient, ClientError
from spark_broker.server import Broker, BrokerHTTPServer, Config
from tests.helpers import request


class StartupCleanupProbe:
    def __init__(self) -> None:
        self.calls = 0

    def startup_cleanup(self) -> None:
        self.calls += 1


class HTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        config = Config(
            broker_id="spark.test", bind="127.0.0.1", port=0, token="t" * 32,
            data_root=Path(self.temporary.name), hunyuan_root=None, stop_containers=(), max_artifact_bytes=1024 * 1024,
        )
        self.broker = Broker(config)
        self.server = BrokerHTTPServer((config.bind, 0), self.broker)
        self.broker.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.client = BrokerClient(self.url, config.token, timeout=5)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.broker.stop()
        self.thread.join(5)
        self.temporary.cleanup()

    def test_auth_is_required_except_liveness(self) -> None:
        with urllib.request.urlopen(f"{self.url}/health/live") as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(f"{self.url}/v1/capabilities")
        self.assertEqual(context.exception.code, 401)
        context.exception.close()

    def test_submit_replay_status_and_events(self) -> None:
        value = request(idempotencyKey="http-replay", metadata={"hello": "world"})
        first = self.client.submit(value)
        second = self.client.submit(value)
        self.assertEqual(first["id"], second["id"])
        for _ in range(100):
            finished = self.client.job(first["id"])
            if finished["status"] == "completed":
                break
        self.assertEqual(finished["status"], "completed")
        self.assertTrue(self.client.events(first["id"])["events"])

    def test_upload_metadata_download_and_hash(self) -> None:
        payload = b"fake png bytes"
        uploaded = self.client.upload_bytes(payload, kind="image", role="source_image", media_type="image/png", origin="tests")
        artifact = uploaded["artifact"]
        self.assertEqual(artifact["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(self.client.artifact(artifact["id"])["artifact"]["mediaType"], "image/png")
        self.assertEqual(self.client.download_bytes(artifact["id"]), payload)
        chunk, headers = self.client.download_chunk(artifact["id"], offset=5, length=3)
        self.assertEqual(chunk, payload[5:8])
        self.assertEqual(headers["Content-Range"], f"bytes 5-7/{len(payload)}")

    def test_capability_and_protocol_skew_fail_without_queuing(self) -> None:
        with self.assertRaises(ClientError) as context:
            self.client.submit(request(capability="image.generate"))
        self.assertEqual(context.exception.code, "capability_unavailable")
        with self.assertRaises(ClientError) as context:
            self.client.submit(request(protocolVersion="99"))
        self.assertEqual(context.exception.code, "unsupported_protocol")


class BrokerStartupTests(unittest.TestCase):
    def test_broker_cleans_executor_crash_residue_before_scheduler_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                broker_id="spark.test", bind="127.0.0.1", port=0, token="t" * 32,
                data_root=Path(directory), hunyuan_root=None, stop_containers=(), max_artifact_bytes=1024 * 1024,
            )
            broker = Broker(config)
            probe = StartupCleanupProbe()
            broker.executors = {"probe": probe}  # type: ignore[assignment]
            broker.scheduler.executors = broker.executors
            broker.start()
            try:
                self.assertEqual(probe.calls, 1)
            finally:
                broker.stop()


if __name__ == "__main__":
    unittest.main()
