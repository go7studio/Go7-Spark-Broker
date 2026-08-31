from __future__ import annotations

import subprocess
import tomllib
import unittest
import re
import json
from pathlib import Path

from spark_broker.resource_probe import ProbePolicy
from spark_broker.resources import ResourcePolicy


ROOT = Path(__file__).parents[1]


class PublicRepositoryTests(unittest.TestCase):
    def test_repository_contains_no_local_identity_or_machine_paths(self) -> None:
        forbidden = (
            re.compile(r"/(?:Users|home)/[^/\s]+/"),
            re.compile(r"\b10(?:\.\d{1,3}){3}\b"),
            re.compile(r"\b192\.168(?:\.\d{1,3}){2}\b"),
            re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b"),
            re.compile(r"\b(?:ssh-rsa|ssh-ed25519)\s+[A-Za-z0-9+/]{32,}"),
            re.compile(r"\b(?:gh[opusr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
        )
        readable = {".md", ".py", ".sh", ".toml", ".yml", ".yaml", ".json", ".service", ".example", ".gitignore"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or any(part in {".git", ".pytest_cache", "__pycache__"} for part in path.parts):
                continue
            if path.suffix not in readable and path.name != "LICENSE":
                continue
            content = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertIsNone(pattern.search(content), f"{path.relative_to(ROOT)} contains private or identifying data matching {pattern.pattern}")

    def test_reachable_history_uses_the_organization_identity(self) -> None:
        history = subprocess.run(
            ["git", "log", "--format=%an%x00%ae"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        for row in history:
            name, _separator, email = row.partition("\0")
            self.assertEqual(name, "Go7 Studio")
            self.assertTrue(email.endswith("@go7.studio"))

    def test_reachable_history_has_no_secret_material(self) -> None:
        patterns = (
            "BEGIN " + "PRIVATE KEY",
            "BEGIN OPENSSH " + "PRIVATE KEY",
            "AKIA" + "[A-Z0-9]{16}",
            "xox" + "[abprs]-[A-Za-z0-9-]{20,}",
            "hf_" + "[A-Za-z0-9]{24,}",
            "gh" + "[opusr]_[A-Za-z0-9]{20,}",
            "sk-" + "[A-Za-z0-9_-]{20,}",
        )
        commits = subprocess.run(
            ["git", "rev-list", "--all"], cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        for commit in commits:
            entries = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit],
                cwd=ROOT, check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            for relative in entries:
                if relative == "tests/test_public_repo.py":
                    continue
                blob = subprocess.run(
                    ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                    check=True, capture_output=True,
                ).stdout.decode("utf-8", errors="ignore")
                for pattern in patterns:
                    self.assertIsNone(
                        re.search(pattern, blob),
                        f"{commit}:{relative} matches secret pattern {pattern}",
                    )

    def test_public_metadata_and_safety_files_ship(self) -> None:
        for relative in (
            "LICENSE",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "ARCHITECTURE.md",
            ".gitignore",
            ".github/workflows/ci.yml",
            "systemd/go7-spark-broker.env.example",
            "examples/inference-routes.example.json",
            "examples/resource-policy.example.json",
            "examples/resource-probe.example.json",
            "examples/controller-state.example.json",
            "examples/training-controller.example.json",
            "examples/checkpoint-receipt.example.json",
            "systemd/go7-spark-training-controller@.service",
            "docs/RESOURCE-GOVERNOR.md",
            "docs/RESOURCE-PROBE.md",
            "docs/TRAINING-INTEGRATION.md",
            "docs/STAGED-ROLLOUT.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["authors"], [{"name": "Go7 Studio"}])
        self.assertEqual(project["urls"]["Repository"], "https://github.com/go7studio/Go7-Spark-Broker.git")

    def test_installer_is_valid_and_does_not_seed_a_model_family(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "deploy-user.sh")], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "deploy-canary-user.sh")], check=True)
        self.assertTrue((ROOT / "deploy-user.sh").stat().st_mode & 0o111)
        self.assertTrue((ROOT / "deploy-canary-user.sh").stat().st_mode & 0o111)
        installer = (ROOT / "deploy-user.sh").read_text(encoding="utf-8").lower()
        canary = (ROOT / "deploy-canary-user.sh").read_text(encoding="utf-8").lower()
        self.assertNotIn("qwen", installer)
        self.assertNotIn("hunyuan", installer)
        self.assertNotIn("qwen", canary)
        self.assertNotIn("hunyuan", canary)
        self.assertIn("inline secrets are refused", installer)
        self.assertIn("elif [[ ! -f \"$broker_config/env\" ]]", installer)
        self.assertIn("systemctl --user restart go7-spark-broker.service", installer)
        self.assertIn("releases/$release_id", installer)
        self.assertIn("health/ready", installer)
        self.assertIn("restoring the previous release", installer)
        self.assertIn("no config, unit, current pointer, database, or running service was changed", installer)
        installer_rollback = installer.split("rollback()", 1)[1].split("on_error()", 1)[0]
        canary_rollback = canary.split("rollback()", 1)[1].split("on_error()", 1)[0]
        self.assertNotIn("sqlite3.connect", installer_rollback)
        self.assertNotIn("sqlite3.connect", canary_rollback)
        self.assertIn("releases/$release_id", canary)
        self.assertIn("health/ready", canary)
        self.assertIn("rollback", canary)
        self.assertIn("no config, unit, current pointer, database, or running service was changed", canary)

    def test_shipped_json_examples_are_current_and_strict(self) -> None:
        routes = json.loads((ROOT / "examples/inference-routes.example.json").read_text(encoding="utf-8"))
        self.assertEqual(routes["version"], 1)
        self.assertTrue(routes["routes"])
        policy = ResourcePolicy.from_value(json.loads(
            (ROOT / "examples/resource-policy.example.json").read_text(encoding="utf-8")
        ))
        self.assertTrue(policy.require_probe)
        self.assertTrue(policy.enforce_memory_admission)
        self.assertTrue(policy.enforce_cuda_admission)
        self.assertEqual(policy.controllers[0].workload_kind, "training")
        self.assertTrue(policy.controllers[0].requires_checkpoint)
        self.assertEqual(policy.controllers[0].timeout_seconds, 600)
        probe = ProbePolicy.from_bytes(
            (ROOT / "examples/resource-probe.example.json").read_bytes()
        )
        self.assertEqual(probe.controller_state_files[0].id, policy.controllers[0].id)
        controller_state = json.loads(
            (ROOT / "examples/controller-state.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(controller_state["controllerId"], policy.controllers[0].id)
        self.assertRegex(controller_state["mutationId"], r"^mutation_[a-f0-9]{32}$")
        training_controller = json.loads(
            (ROOT / "examples/training-controller.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(training_controller["controllerId"], policy.controllers[0].id)
        self.assertEqual(training_controller["profileId"], policy.controllers[0].profile_id)
        self.assertEqual(training_controller["releasedMode"], policy.controllers[0].throttled_mode)
        self.assertEqual(
            training_controller["stateFile"],
            str(probe.controller_state_files[0].path),
        )
        receipt = json.loads(
            (ROOT / "examples/checkpoint-receipt.example.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["version"], 1)
        self.assertRegex(receipt["files"][0]["sha256"], r"^[a-f0-9]{64}$")


if __name__ == "__main__":
    unittest.main()
