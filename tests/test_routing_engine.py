from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from spark_broker.routing import (
    RoutingError,
    compile_route_policies,
    compile_routing_config,
    simulate_routing_scenarios,
)


ROOT = Path(__file__).parents[1]
SCENARIOS = ROOT / "tests" / "scenarios"


def routing_config() -> dict:
    return {
        "version": 1,
        "routes": [
            {
                "id": "small",
                "model": "small-model",
                "profileId": "gpu.small",
                "description": "Small route",
                "endpoint": "http://127.0.0.1:8001",
                "apiKeyFile": "/run/secrets/small-key",
                "container": "small-runtime",
                "estimatedMemoryGb": 24,
                "priority": 90,
                "serviceClasses": ["interactive", "batch"],
            },
            {
                "id": "large",
                "model": "large-model",
                "profileId": "gpu.large",
                "description": "Large route",
                "endpoint": "http://localhost:8002/",
                "apiKeyFile": "/run/secrets/large-key",
                "container": "large-runtime",
                "estimatedMemoryGb": 72,
                "priority": 60,
                "serviceClasses": ["interactive", "batch", "background"],
            },
        ],
    }


class RoutingCompilerTests(unittest.TestCase):
    def test_revision_is_canonical_for_semantically_identical_ordering(self) -> None:
        first = routing_config()
        second = copy.deepcopy(first)
        second["routes"].reverse()
        second["routes"][0]["serviceClasses"].reverse()
        self.assertEqual(
            compile_routing_config(first).revision,
            compile_routing_config(second).revision,
        )
        self.assertEqual(len(compile_routing_config(first).revision), 64)

    def test_runtime_policy_revision_cannot_be_replaced_by_a_declared_digest(self) -> None:
        values = [{
            "id": "small", "model": "model-small", "profileId": "gpu.small",
            "estimatedMemoryGb": 20, "priority": 80,
            "serviceClasses": ["interactive"],
        }]
        compiled = compile_route_policies(values)
        self.assertEqual(compile_route_policies(values, declared_revision=compiled.revision), compiled)
        with self.assertRaisesRegex(RoutingError, "does not match"):
            compile_route_policies(values, declared_revision="0" * 64)

    def test_compiler_rejects_weakly_typed_or_ambiguous_routes(self) -> None:
        mutations = [
            ("priority", True),
            ("estimatedMemoryGb", "24"),
            ("serviceClasses", ["interactive", "interactive"]),
            ("profileId", "not-a-profile"),
            ("endpoint", "https://public.example.invalid"),
            ("container", "bad container"),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                config = routing_config()
                config["routes"][0][key] = value
                with self.assertRaises(RoutingError):
                    compile_routing_config(config)
        config = routing_config()
        config["routes"][0]["typoPriority"] = 10
        with self.assertRaisesRegex(RoutingError, "unknown fields"):
            compile_routing_config(config)

    def test_all_preferences_and_route_permutations_are_deterministic(self) -> None:
        first = compile_routing_config(routing_config()).engine()
        reversed_config = routing_config()
        reversed_config["routes"].reverse()
        second = compile_routing_config(reversed_config).engine()
        contexts = {
            "balanced": {
                "activeProfiles": ["gpu.large"],
                "profiles": {"gpu.large": {"health": "healthy"}},
            },
            "latency": {
                "profiles": {"gpu.small": {"latencyMs": 90}, "gpu.large": {"latencyMs": 20}},
            },
            "throughput": {
                "profiles": {"gpu.small": {"availableConcurrency": 1}, "gpu.large": {"availableConcurrency": 4}},
            },
            "memory": {},
        }
        expected = {"balanced": "large", "latency": "large", "throughput": "large", "memory": "small"}
        for preference, snapshot in contexts.items():
            with self.subTest(preference=preference):
                left = first.decide(preference=preference, snapshot=snapshot)
                right = second.decide(preference=preference, snapshot=snapshot)
                self.assertEqual(left.route_id, expected[preference])
                self.assertEqual(left.public(), right.public())

    def test_route_metrics_are_strictly_typed_and_finite(self) -> None:
        engine = compile_routing_config(routing_config()).engine()
        invalid_profiles = [
            {"gpu.small": {"latencyMs": "fast"}},
            {"gpu.small": {"latencyMs": float("nan")}},
            {"gpu.small": {"latencyMs": float("inf")}},
            {"gpu.small": {"latencyMs": -1}},
            {"gpu.small": {"availableConcurrency": True}},
            {"gpu.small": {"availableConcurrency": 1.5}},
            {"gpu.small": {"health": "mostly-healthy"}},
        ]
        for profiles in invalid_profiles:
            with self.subTest(profiles=profiles):
                with self.assertRaises(RoutingError) as context:
                    engine.decide(snapshot={"profiles": profiles})
                self.assertEqual(context.exception.code, "invalid_metric")


class RoutingSimulationTests(unittest.TestCase):
    def test_golden_scenario(self) -> None:
        scenario = json.loads((SCENARIOS / "routing-basic.json").read_text(encoding="utf-8"))
        expected = json.loads((SCENARIOS / "routing-basic.expected.json").read_text(encoding="utf-8"))
        self.assertEqual(simulate_routing_scenarios(scenario), expected)

    def test_cli_simulation_is_offline_and_needs_no_broker_token(self) -> None:
        environment = {**os.environ, "PYTHONPATH": str(ROOT)}
        environment.pop("SPARK_BROKER_TOKEN", None)
        environment.pop("SPARK_BROKER_TOKEN_FILE", None)
        completed = subprocess.run(
            [sys.executable, "-m", "spark_broker.cli", "route-simulate", str(SCENARIOS / "routing-basic.json")],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        self.assertEqual(json.loads(completed.stdout), simulate_routing_scenarios(
            json.loads((SCENARIOS / "routing-basic.json").read_text(encoding="utf-8"))
        ))


if __name__ == "__main__":
    unittest.main()
