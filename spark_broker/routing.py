from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_ROUTE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9][a-z0-9-]*){1,7}$")
_CONTAINER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SCENARIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REVISION = re.compile(r"^[a-f0-9]{64}$")
_SERVICE_CLASSES = {"interactive", "batch", "background"}
_PREFERENCES = {"balanced", "latency", "throughput", "memory"}
_HEALTH_VALUES = {"healthy", "unhealthy", "degraded", "unknown"}


class RoutingError(ValueError):
    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.field is not None:
            value["field"] = self.field
        return value


@dataclass(frozen=True)
class RoutePolicy:
    id: str
    model: str
    profile_id: str
    estimated_memory_gb: int
    priority: int
    service_classes: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "profileId": self.profile_id,
            "estimatedMemoryGb": self.estimated_memory_gb,
            "priority": self.priority,
            "serviceClasses": list(self.service_classes),
        }


@dataclass(frozen=True)
class CompiledRoutingConfig:
    version: int
    routes: tuple[RoutePolicy, ...]
    revision: str
    canonical_json: str

    def engine(self) -> "RoutingEngine":
        return RoutingEngine(self)

    def public(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "revision": self.revision,
            "routeCount": len(self.routes),
            "routeIds": [route.id for route in self.routes],
        }


@dataclass(frozen=True)
class RouteDecision:
    route_id: str
    profile_id: str
    model: str
    reason: str
    preference: str
    config_revision: str
    ranked_route_ids: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {
            "routeId": self.route_id,
            "profileId": self.profile_id,
            "model": self.model,
            "reason": self.reason,
            "preference": self.preference,
            "configRevision": self.config_revision,
            "rankedRouteIds": list(self.ranked_route_ids),
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _revision(value: Any) -> tuple[str, str]:
    canonical = _canonical_json(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), canonical


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RoutingError("invalid_type", f"{field} must be an object", field=field)
    return value


def _string(value: Any, field: str, *, maximum: int, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise RoutingError(
            "invalid_type",
            f"{field} must be a non-empty string up to {maximum} characters",
            field=field,
        )
    if pattern is not None and not pattern.fullmatch(value):
        raise RoutingError("invalid_format", f"{field} has an invalid format", field=field)
    return value


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise RoutingError(
            "invalid_type",
            f"{field} must be an integer from {minimum} through {maximum}",
            field=field,
        )
    return value


def _route_policy(item: Mapping[str, Any], field: str) -> RoutePolicy:
    route_id = _string(item.get("id"), f"{field}.id", maximum=128, pattern=_ROUTE_ID)
    model = _string(item.get("model"), f"{field}.model", maximum=256)
    profile_id = _string(
        item.get("profileId", f"gpu.{route_id}"),
        f"{field}.profileId",
        maximum=128,
        pattern=_PROFILE_ID,
    )
    memory = _integer(item.get("estimatedMemoryGb", 0), f"{field}.estimatedMemoryGb", minimum=0, maximum=1024)
    priority = _integer(item.get("priority", 50), f"{field}.priority", minimum=0, maximum=100)
    raw_classes = item.get("serviceClasses", ["interactive", "batch"])
    if not isinstance(raw_classes, list) or not raw_classes or len(raw_classes) > len(_SERVICE_CLASSES):
        raise RoutingError(
            "invalid_type",
            f"{field}.serviceClasses must contain 1-{len(_SERVICE_CLASSES)} entries",
            field=f"{field}.serviceClasses",
        )
    if any(not isinstance(value, str) or value not in _SERVICE_CLASSES for value in raw_classes):
        raise RoutingError(
            "invalid_value",
            f"{field}.serviceClasses contains an invalid service class",
            field=f"{field}.serviceClasses",
        )
    if len(raw_classes) != len(set(raw_classes)):
        raise RoutingError(
            "duplicate_value",
            f"{field}.serviceClasses must not contain duplicates",
            field=f"{field}.serviceClasses",
        )
    return RoutePolicy(
        id=route_id,
        model=model,
        profile_id=profile_id,
        estimated_memory_gb=memory,
        priority=priority,
        service_classes=tuple(sorted(raw_classes)),
    )


def _validate_route_set(routes: Iterable[RoutePolicy]) -> tuple[RoutePolicy, ...]:
    result = tuple(sorted(routes, key=lambda route: route.id))
    if not result or len(result) > 32:
        raise RoutingError("invalid_routes", "routes must contain 1-32 entries", field="routes")
    if len({route.id for route in result}) != len(result):
        raise RoutingError("duplicate_route", "route ids must be unique", field="routes")
    if len({route.profile_id for route in result}) != len(result):
        raise RoutingError("duplicate_profile", "route profile ids must be unique", field="routes")
    return result


def _loopback_url(value: Any, field: str) -> str:
    text = _string(value, field, maximum=2048)
    try:
        parsed = urllib.parse.urlsplit(text)
        parsed.port
    except ValueError as exc:
        raise RoutingError("invalid_url", f"{field} must be a valid HTTP loopback URL", field=field) from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RoutingError(
            "invalid_url",
            f"{field} must be an HTTP loopback URL without credentials, query, or fragment",
            field=field,
        )
    return text.rstrip("/")


def compile_routing_config(value: Any) -> CompiledRoutingConfig:
    """Compile and fingerprint an administrator routing document without I/O."""

    root = _object(value, "config")
    if set(root) != {"version", "routes"} or root.get("version") != 1:
        raise RoutingError(
            "invalid_config",
            "routing config must contain exactly version=1 and routes",
            field="config",
        )
    raw_routes = root["routes"]
    if not isinstance(raw_routes, list) or not 1 <= len(raw_routes) <= 32:
        raise RoutingError("invalid_routes", "routes must contain 1-32 entries", field="routes")
    allowed = {
        "id",
        "model",
        "profileId",
        "description",
        "endpoint",
        "apiKeyFile",
        "container",
        "estimatedMemoryGb",
        "priority",
        "serviceClasses",
    }
    policies: list[RoutePolicy] = []
    canonical_routes: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_routes):
        field = f"routes[{index}]"
        item = _object(raw_item, field)
        unknown = set(item) - allowed
        if unknown:
            raise RoutingError(
                "unknown_field",
                f"{field} contains unknown fields: {sorted(unknown)}",
                field=field,
            )
        policy = _route_policy(item, field)
        description = item.get("description", f"Local model route {policy.id}")
        if not isinstance(description, str) or not description.strip() or len(description) > 256:
            raise RoutingError(
                "invalid_type",
                f"{field}.description must be a non-empty string up to 256 characters",
                field=f"{field}.description",
            )
        endpoint = _loopback_url(item.get("endpoint"), f"{field}.endpoint")
        api_key_file = _string(item.get("apiKeyFile"), f"{field}.apiKeyFile", maximum=4096)
        if not Path(api_key_file).is_absolute():
            raise RoutingError(
                "invalid_path",
                f"{field}.apiKeyFile must be an absolute path",
                field=f"{field}.apiKeyFile",
            )
        container = item.get("container")
        if container is not None:
            container = _string(container, f"{field}.container", maximum=128, pattern=_CONTAINER)
        policies.append(policy)
        canonical_routes.append({
            **policy.canonical(),
            "description": description.strip(),
            "endpoint": endpoint,
            "apiKeyFile": api_key_file,
            "container": container,
        })
    compiled_routes = _validate_route_set(policies)
    canonical_value = {
        "version": 1,
        "routes": sorted(canonical_routes, key=lambda item: item["id"]),
    }
    digest, canonical = _revision(canonical_value)
    return CompiledRoutingConfig(version=1, routes=compiled_routes, revision=digest, canonical_json=canonical)


def compile_route_policies(
    values: Iterable[Mapping[str, Any]], *, declared_revision: str | None = None
) -> CompiledRoutingConfig:
    """Compile already-secret-resolved runtime routes for the production selector."""

    policies: list[RoutePolicy] = []
    allowed = {"id", "model", "profileId", "estimatedMemoryGb", "priority", "serviceClasses"}
    for index, item in enumerate(values):
        field = f"routes[{index}]"
        unknown = set(item) - allowed
        if unknown:
            raise RoutingError("unknown_field", f"{field} contains unknown fields: {sorted(unknown)}", field=field)
        policies.append(_route_policy(item, field))
    routes = _validate_route_set(policies)
    canonical_value = {"version": 1, "routes": [route.canonical() for route in routes]}
    computed, canonical = _revision(canonical_value)
    if declared_revision is not None:
        if not isinstance(declared_revision, str) or not _REVISION.fullmatch(declared_revision):
            raise RoutingError("invalid_revision", "declared routing revision is invalid", field="revision")
        if declared_revision != computed:
            raise RoutingError(
                "invalid_revision",
                "declared routing revision does not match the canonical route policies",
                field="revision",
            )
    return CompiledRoutingConfig(version=1, routes=routes, revision=computed, canonical_json=canonical)


def _metric_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RoutingError("invalid_metric", f"{field} must be a finite non-negative number", field=field)
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RoutingError("invalid_metric", f"{field} must be a finite non-negative number", field=field)
    return result


def normalize_routing_snapshot(value: Any, *, reject_unknown: bool = False) -> dict[str, Any]:
    if value is None:
        value = {}
    snapshot = _object(value, "snapshot")
    allowed = {"probeHealthy", "activeProfiles", "profiles"}
    if reject_unknown and set(snapshot) - allowed:
        unknown = sorted(set(snapshot) - allowed)
        raise RoutingError("unknown_field", f"snapshot contains unknown fields: {unknown}", field="snapshot")
    if "probeHealthy" in snapshot and snapshot["probeHealthy"] is not None and not isinstance(snapshot["probeHealthy"], bool):
        raise RoutingError("invalid_metric", "snapshot.probeHealthy must be boolean or null", field="snapshot.probeHealthy")
    active = snapshot.get("activeProfiles", [])
    if not isinstance(active, list) or any(not isinstance(item, str) or not item for item in active):
        raise RoutingError("invalid_metric", "snapshot.activeProfiles must be a list of profile ids", field="snapshot.activeProfiles")
    if len(active) != len(set(active)):
        raise RoutingError("duplicate_value", "snapshot.activeProfiles must not contain duplicates", field="snapshot.activeProfiles")
    profiles = snapshot.get("profiles", {})
    if not isinstance(profiles, dict) or any(not isinstance(key, str) or not key for key in profiles):
        raise RoutingError("invalid_metric", "snapshot.profiles must be an object keyed by profile id", field="snapshot.profiles")
    clean_profiles: dict[str, dict[str, Any]] = {}
    for profile_id, raw_profile in profiles.items():
        field = f"snapshot.profiles.{profile_id}"
        profile = _object(raw_profile, field)
        metric_fields = {"health", "latencyMs", "availableConcurrency"}
        if reject_unknown and set(profile) - metric_fields:
            unknown = sorted(set(profile) - metric_fields)
            raise RoutingError(
                "unknown_field",
                f"{field} contains unknown metric fields: {unknown}",
                field=field,
            )
        clean: dict[str, Any] = {}
        if "health" in profile:
            health = profile["health"]
            if not isinstance(health, str) or health not in _HEALTH_VALUES:
                raise RoutingError("invalid_metric", f"{field}.health is invalid", field=f"{field}.health")
            clean["health"] = health
        if "latencyMs" in profile:
            clean["latencyMs"] = _metric_number(profile["latencyMs"], f"{field}.latencyMs")
        if "availableConcurrency" in profile:
            concurrency = profile["availableConcurrency"]
            if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 0:
                raise RoutingError(
                    "invalid_metric",
                    f"{field}.availableConcurrency must be a non-negative integer",
                    field=f"{field}.availableConcurrency",
                )
            clean["availableConcurrency"] = concurrency
        clean_profiles[profile_id] = clean
    return {
        "probeHealthy": snapshot.get("probeHealthy"),
        "activeProfiles": list(active),
        "profiles": clean_profiles,
    }


class RoutingEngine:
    def __init__(self, config: CompiledRoutingConfig) -> None:
        self.config = config

    def decide(
        self,
        *,
        model: str | None = None,
        service_class: str = "interactive",
        preference: str = "balanced",
        snapshot: Any = None,
    ) -> RouteDecision:
        if model is not None and (not isinstance(model, str) or not model):
            raise RoutingError("invalid_request", "model must be a non-empty string", field="model")
        if service_class not in _SERVICE_CLASSES:
            raise RoutingError("invalid_request", "serviceClass is invalid", field="serviceClass")
        if preference not in _PREFERENCES:
            raise RoutingError("invalid_request", "routePreference is invalid", field="routePreference")
        if model is not None and model not in {route.model for route in self.config.routes}:
            raise RoutingError("route_unavailable", "the requested model is not installed", field="model")
        context = normalize_routing_snapshot(snapshot)
        candidates = [
            route
            for route in self.config.routes
            if (model is None or route.model == model) and service_class in route.service_classes
        ]
        if not candidates:
            raise RoutingError(
                "route_unavailable",
                "no installed inference profile satisfies the request",
                field="request",
            )
        profiles = context["profiles"]
        candidates = [
            route for route in candidates if profiles.get(route.profile_id, {}).get("health") != "unhealthy"
        ]
        if not candidates:
            raise RoutingError("route_unavailable", "all matching inference profiles are unhealthy", field="request")
        active_profiles = set(context["activeProfiles"])
        if preference == "memory":
            candidates.sort(key=lambda route: (route.estimated_memory_gb, -route.priority, route.id))
        elif preference == "latency":
            candidates.sort(key=lambda route: (
                profiles.get(route.profile_id, {}).get("latencyMs", float("inf")),
                -route.priority,
                route.id,
            ))
        elif preference == "throughput":
            candidates.sort(key=lambda route: (
                -profiles.get(route.profile_id, {}).get("availableConcurrency", 0),
                -route.priority,
                route.id,
            ))
        else:
            candidates.sort(key=lambda route: (
                route.profile_id not in active_profiles,
                -route.priority,
                route.estimated_memory_gb,
                route.id,
            ))
        selected = candidates[0]
        return RouteDecision(
            route_id=selected.id,
            profile_id=selected.profile_id,
            model=selected.model,
            reason="explicit_model" if model is not None else f"policy:{preference}",
            preference=preference,
            config_revision=self.config.revision,
            ranked_route_ids=tuple(route.id for route in candidates),
        )


def simulate_routing_scenarios(value: Any) -> dict[str, Any]:
    """Evaluate routing cases as pure data; no credentials, runtimes, or stores are touched."""

    root = _object(value, "scenario")
    if set(root) != {"version", "config", "cases"} or root.get("version") != 1:
        raise RoutingError(
            "invalid_scenario",
            "scenario must contain exactly version=1, config, and cases",
            field="scenario",
        )
    config = compile_routing_config(root["config"])
    raw_cases = root["cases"]
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= 1000:
        raise RoutingError("invalid_scenario", "scenario cases must contain 1-1000 entries", field="cases")
    case_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        field = f"cases[{index}]"
        case = _object(raw_case, field)
        if set(case) != {"id", "request", "snapshot"}:
            raise RoutingError(
                "invalid_scenario",
                f"{field} must contain exactly id, request, and snapshot",
                field=field,
            )
        case_id = _string(case["id"], f"{field}.id", maximum=128, pattern=_SCENARIO_ID)
        if case_id in case_ids:
            raise RoutingError("duplicate_value", "scenario case ids must be unique", field=f"{field}.id")
        case_ids.add(case_id)
        request = _object(case["request"], f"{field}.request")
        allowed_request = {"model", "serviceClass", "routePreference"}
        if set(request) - allowed_request:
            raise RoutingError(
                "unknown_field",
                f"{field}.request contains unknown fields: {sorted(set(request) - allowed_request)}",
                field=f"{field}.request",
            )
        try:
            snapshot = normalize_routing_snapshot(case["snapshot"], reject_unknown=True)
            decision = config.engine().decide(
                model=request.get("model"),
                service_class=request.get("serviceClass", "interactive"),
                preference=request.get("routePreference", "balanced"),
                snapshot=snapshot,
            )
        except RoutingError as exc:
            results.append({"id": case_id, "status": "rejected", "error": exc.as_dict()})
        else:
            results.append({"id": case_id, "status": "selected", "decision": decision.public()})
    return {"version": 1, "configRevision": config.revision, "cases": results}
