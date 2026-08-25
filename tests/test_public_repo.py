from __future__ import annotations

import subprocess
import tomllib
import unittest
import re
import json
from pathlib import Path

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
            "docs/RESOURCE-GOVERNOR.md",
            "docs/TRAINING-INTEGRATION.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["authors"], [{"name": "Go7 Studio"}])
        self.assertEqual(project["urls"]["Repository"], "https://github.com/go7studio/Go7-Spark-Broker.git")

    def test_installer_is_valid_and_does_not_seed_a_model_family(self) -> None:
        subprocess.run(["bash", "-n", str(ROOT / "deploy-user.sh")], check=True)
        installer = (ROOT / "deploy-user.sh").read_text(encoding="utf-8").lower()
        self.assertNotIn("qwen", installer)
        self.assertNotIn("hunyuan", installer)
        self.assertIn("inline secrets are refused", installer)
        self.assertIn("elif [[ ! -f \"$broker_config/env\" ]]", installer)
        self.assertIn("systemctl --user restart go7-spark-broker.service", installer)

    def test_shipped_json_examples_are_current_and_strict(self) -> None:
        routes = json.loads((ROOT / "examples/inference-routes.example.json").read_text(encoding="utf-8"))
        self.assertEqual(routes["version"], 1)
        self.assertTrue(routes["routes"])
        policy = ResourcePolicy.from_file(ROOT / "examples/resource-policy.example.json")
        self.assertTrue(policy.require_probe)
        self.assertTrue(policy.enforce_memory_admission)
        self.assertEqual(policy.controllers[0].workload_kind, "training")
        self.assertTrue(policy.controllers[0].requires_checkpoint)


if __name__ == "__main__":
    unittest.main()
