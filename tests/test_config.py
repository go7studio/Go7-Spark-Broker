from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spark_broker.server import Config


class ConfigTests(unittest.TestCase):
    @staticmethod
    def credential(directory: str, name: str, value: str) -> str:
        path = Path(directory) / name
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return str(path)

    def test_minimal_configuration_has_no_model_or_container_assumptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = self.credential(directory, "broker-token", "t" * 32)
            with patch.dict(os.environ, {
                "SPARK_BROKER_TOKEN_FILE": token,
                "SPARK_BROKER_DATA": directory,
            }, clear=True):
                config = Config.from_env()
        self.assertEqual(config.broker_id, "local-capability-host")
        self.assertEqual(config.stop_containers, ())
        self.assertIsNone(config.text_endpoint)
        self.assertEqual(config.text_model, "")
        self.assertEqual(config.text_profile_id, "gpu.openai-compatible")
        self.assertEqual(config.text_estimated_memory_gb, 0)
        self.assertIsNone(config.coordinator_lock_file)

    def test_host_coordinator_lock_path_is_configurable_independently_of_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = self.credential(directory, "broker-token", "t" * 32)
            lock_file = Path(directory) / "host-gpu0.lock"
            epoch_file = Path(directory) / "host-gpu0.epoch"
            with patch.dict(os.environ, {
                "SPARK_BROKER_TOKEN_FILE": token,
                "SPARK_BROKER_DATA": str(Path(directory) / "data"),
                "SPARK_COORDINATOR_LOCK_FILE": str(lock_file),
                "SPARK_COORDINATOR_EPOCH_FILE": str(epoch_file),
            }, clear=True):
                config = Config.from_env()
        self.assertEqual(config.coordinator_lock_file, lock_file)
        self.assertEqual(config.coordinator_epoch_file, epoch_file)

    def test_generic_openai_configuration_is_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            key = self.credential(directory, "runtime-key", "runtime-secret")
            token = self.credential(directory, "broker-token", "t" * 32)
            with patch.dict(os.environ, {
                "SPARK_BROKER_TOKEN_FILE": token,
                "SPARK_BROKER_DATA": directory,
                "SPARK_OPENAI_ENDPOINT": "http://127.0.0.1:8000",
                "SPARK_OPENAI_API_KEY_FILE": key,
                "SPARK_OPENAI_MODEL": "local-model",
                "SPARK_OPENAI_CONTAINER": "local-runtime",
                "SPARK_OPENAI_PROFILE_ID": "gpu.local-model",
                "SPARK_OPENAI_DESCRIPTION": "Configured test runtime",
                "SPARK_OPENAI_ESTIMATED_MEMORY_GB": "64",
            }, clear=True):
                config = Config.from_env()
        self.assertEqual(config.text_endpoint, "http://127.0.0.1:8000")
        self.assertEqual(config.text_api_key, "runtime-secret")
        self.assertEqual(config.text_model, "local-model")
        self.assertEqual(config.text_container, "local-runtime")
        self.assertEqual(config.text_profile_id, "gpu.local-model")
        self.assertEqual(config.text_description, "Configured test runtime")
        self.assertEqual(config.text_estimated_memory_gb, 64)

    def test_legacy_text_variables_remain_an_upgrade_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = self.credential(directory, "broker-token", "t" * 32)
            runtime = self.credential(directory, "runtime-key", "legacy-runtime-secret")
            with patch.dict(os.environ, {
                "SPARK_BROKER_TOKEN_FILE": token,
                "SPARK_BROKER_DATA": directory,
                "SPARK_TEXT_ENDPOINT": "http://127.0.0.1:8001",
                "SPARK_TEXT_API_KEY_FILE": runtime,
                "SPARK_TEXT_MODEL": "legacy-model",
                "SPARK_TEXT_CONTAINER": "legacy-runtime",
            }, clear=True):
                config = Config.from_env()
        self.assertEqual(config.text_endpoint, "http://127.0.0.1:8001")
        self.assertEqual(config.text_api_key, "legacy-runtime-secret")
        self.assertEqual(config.text_model, "legacy-model")
        self.assertEqual(config.text_container, "legacy-runtime")

    def test_partial_text_configuration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token = self.credential(directory, "broker-token", "t" * 32)
            with patch.dict(os.environ, {
                "SPARK_BROKER_TOKEN_FILE": token,
                "SPARK_BROKER_DATA": directory,
                "SPARK_OPENAI_ENDPOINT": "http://127.0.0.1:8000",
                "SPARK_OPENAI_MODEL": "missing-credential",
            }, clear=True):
                with self.assertRaisesRegex(SystemExit, "endpoint, credential, and model together"):
                    Config.from_env()

    def test_inline_secrets_and_loose_credential_modes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SPARK_BROKER_TOKEN": "t" * 32}, clear=True):
                with self.assertRaisesRegex(SystemExit, "inline broker secrets are refused"):
                    Config.from_env()
            token = Path(directory) / "loose-token"
            token.write_text("t" * 32, encoding="utf-8")
            token.chmod(0o644)
            with patch.dict(os.environ, {"SPARK_BROKER_TOKEN_FILE": str(token)}, clear=True):
                with self.assertRaisesRegex(SystemExit, "0600"):
                    Config.from_env()


if __name__ == "__main__":
    unittest.main()
