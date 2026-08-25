from __future__ import annotations

import unittest

from spark_broker.contract import ContractError, validate_job_request
from tests.helpers import request


class ContractTests(unittest.TestCase):
    def validate(self, value: dict) -> dict:
        return validate_job_request(value, broker_id="spark.test", max_hops=8)

    def test_valid_request_is_normalized(self) -> None:
        value = self.validate(request(metadata={"purpose": "test"}))
        self.assertEqual(value["priority"], 50)
        self.assertEqual(value["workflow"]["maxContinuations"], 0)

    def test_unknown_top_level_field_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "unknown fields"):
            self.validate(request(command="docker rm -f anything"))

    def test_protocol_skew_fails_closed(self) -> None:
        with self.assertRaises(ContractError) as context:
            self.validate(request(protocolVersion="2.0"))
        self.assertEqual(context.exception.code, "unsupported_protocol")

    def test_cycle_and_duplicate_routes_are_rejected(self) -> None:
        with self.assertRaises(ContractError) as context:
            self.validate(request(visitedSystems=["spark.test"], hopCount=1))
        self.assertEqual(context.exception.code, "route_cycle")
        with self.assertRaises(ContractError) as context:
            self.validate(request(visitedSystems=["a", "a"], hopCount=2))
        self.assertEqual(context.exception.code, "route_cycle")

    def test_hop_count_must_match_route_and_stay_below_limit(self) -> None:
        with self.assertRaises(ContractError) as context:
            self.validate(request(visitedSystems=["a"], hopCount=0))
        self.assertEqual(context.exception.code, "invalid_route")
        visited = [f"node-{i}" for i in range(8)]
        with self.assertRaises(ContractError) as context:
            self.validate(request(visitedSystems=visited, hopCount=8))
        self.assertEqual(context.exception.code, "hop_limit")

    def test_output_roles_must_be_unique(self) -> None:
        output = {"role": "model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True}
        with self.assertRaises(ContractError) as context:
            self.validate(request(requiredOutputs=[output, output]))
        self.assertEqual(context.exception.code, "duplicate_output")

    def test_auto_continue_requires_explicit_schema_only(self) -> None:
        with self.assertRaises(ContractError):
            self.validate(request(workflow={"autoContinue": True, "approvedCapabilities": [], "maxContinuations": 1, "shell": "do it"}))


if __name__ == "__main__":
    unittest.main()
