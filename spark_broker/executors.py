from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .artifacts import ArtifactRegistry
from .contract import ContractError
from .resources import ExecutionPlan
from .routing import RoutingError, compile_route_policies, compile_routing_config
from .secure_files import SecureFileError, read_owner_secret, read_owner_text


class ExecutionCancelled(RuntimeError):
    pass


class ExecutionFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True)
class Capability:
    id: str
    profile_id: str
    description: str
    input_kinds: tuple[str, ...]
    output_roles: tuple[str, ...]
    estimated_memory_gb: int
    invocation: dict[str, Any] | None = None
    continuations: tuple[dict[str, Any], ...] = ()
    resource_group: str | None = "gpu:0"
    service_class: str = "batch"
    lease_mode: str = "exclusive"
    preemption_mode: str = "none"

    def public(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "profileId": self.profile_id,
            "description": self.description,
            "inputKinds": list(self.input_kinds),
            "outputRoles": list(self.output_roles),
            "estimatedMemoryGb": self.estimated_memory_gb,
            "asynchronous": True,
            "resourcePolicy": {
                "resourceGroup": self.resource_group,
                "serviceClass": self.service_class,
                "leaseMode": self.lease_mode,
                "preemptionMode": self.preemption_mode,
            },
        }
        if self.invocation is not None:
            value["invocation"] = self.invocation
        if self.continuations:
            value["continuations"] = list(self.continuations)
        return value


class CommandRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: subprocess.Popen[bytes] | None = None

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        log_path: Path,
        timeout: int,
        cancelled: Callable[[], bool],
        cleanup: Callable[[], None] | None = None,
        check: bool = True,
    ) -> int:
        if not argv or argv[0] not in {"docker"}:
            raise ExecutionFailure("unsafe_command", "executor attempted a command outside the allowlist")
        if cancelled():
            raise ExecutionCancelled("job was cancelled")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self._lock:
                self._active = process
            try:
                while process.poll() is None:
                    if cancelled():
                        self._terminate(process)
                        if cleanup:
                            cleanup()
                        raise ExecutionCancelled("job was cancelled")
                    if time.monotonic() - started > timeout:
                        self._terminate(process)
                        if cleanup:
                            cleanup()
                        raise ExecutionFailure("stage_timeout", f"stage exceeded {timeout} seconds", retryable=True)
                    time.sleep(0.25)
                code = int(process.returncode or 0)
            finally:
                with self._lock:
                    if self._active is process:
                        self._active = None
        if check and code != 0:
            raise ExecutionFailure("stage_failed", f"executor stage exited with code {code}", retryable=False)
        return code

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class Executor:
    capability: Capability

    def startup_cleanup(self) -> None:
        """Remove executor-owned crash residue before any queued profile can load."""

    def validate_request(self, job: dict[str, Any], registry: ArtifactRegistry) -> None:
        del job, registry

    def plan(self, job: dict[str, Any], routing_context: dict[str, Any] | None = None) -> ExecutionPlan:
        del job, routing_context
        return ExecutionPlan(
            profile_id=self.capability.profile_id,
            route_id=None,
            resource_group=self.capability.resource_group,
            service_class=self.capability.service_class,
            lease_mode=self.capability.lease_mode,
            estimated_memory_gb=self.capability.estimated_memory_gb,
            preemption_mode=self.capability.preemption_mode,
        )

    def activate_plan(self, plan: ExecutionPlan, cancelled: Callable[[], bool]) -> None:
        del plan
        self.activate(cancelled)

    def execute_plan(
        self,
        plan: ExecutionPlan,
        job: dict[str, Any],
        registry: ArtifactRegistry,
        cancelled: Callable[[], bool],
        stage: Callable[[str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        del plan
        return self.execute(job, registry, cancelled, stage)

    def deactivate_plan(self, plan: ExecutionPlan) -> bool:
        del plan
        return self.capability.resource_group is None

    def has_managed_unload(self) -> bool:
        return self.capability.resource_group is None

    def gpu_profile_lifecycles(self) -> dict[str, bool]:
        if self.capability.resource_group is None:
            return {}
        return {self.capability.profile_id: self.has_managed_unload()}

    def activate(self, cancelled: Callable[[], bool]) -> None:
        del cancelled

    def execute(
        self,
        job: dict[str, Any],
        registry: ArtifactRegistry,
        cancelled: Callable[[], bool],
        stage: Callable[[str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        raise NotImplementedError


class EchoExecutor(Executor):
    capability = Capability(
        id="system.echo",
        profile_id="cpu.echo",
        description="Deterministic broker contract and scheduling check",
        input_kinds=(),
        output_roles=(),
        estimated_memory_gb=0,
        invocation={
            "inputs": [],
            "outputs": [],
            "constraintsSchema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        resource_group=None,
        service_class="system",
        lease_mode="none",
    )

    def execute(self, job: dict[str, Any], registry: ArtifactRegistry, cancelled: Callable[[], bool], stage: Callable[[str, dict[str, Any] | None], None]) -> dict[str, Any]:
        del registry
        if cancelled():
            raise ExecutionCancelled("job was cancelled")
        stage("running", {"executor": "echo"})
        return {"artifacts": [], "continuations": [], "data": {"echo": job["request"].get("metadata", {})}}


class Hunyuan3DExecutor(Executor):
    capability = Capability(
        id="asset.3d.generate",
        profile_id="gpu.hunyuan3d",
        description="Generate a GLB shape or PBR asset from an input image",
        input_kinds=("image",),
        output_roles=("shape_model", "pbr_model", "mesh_report", "pbr_mesh_report", "execution_log"),
        estimated_memory_gb=16,
        invocation={
            "inputs": [{
                "role": "source_image", "kind": "image",
                "mediaTypes": ["image/png", "image/jpeg", "image/webp"],
                "required": True, "minItems": 1, "maxItems": 1,
            }],
            "outputs": [
                {"role": "shape_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
                {"role": "pbr_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": False},
                {"role": "pbr_mesh_report", "kind": "report", "mediaTypes": ["application/json"], "required": False},
                {"role": "execution_log", "kind": "log", "mediaTypes": ["text/plain"], "required": False},
            ],
            "constraintsSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["shape", "pbr"], "default": "shape"},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 42},
                    "textureViews": {"type": "integer", "minimum": 1, "maximum": 16, "default": 6},
                    "textureResolution": {"type": "integer", "minimum": 128, "maximum": 2048, "default": 512},
                    "maxFaces": {"type": "integer", "minimum": 100, "maximum": 10000000, "default": 1000000},
                    "targetEngine": {"type": "string", "enum": ["generic", "blender", "godot", "unity", "unreal"], "default": "generic"},
                    "units": {"type": "string", "enum": ["meters", "centimeters"], "default": "meters"},
                    "upAxis": {"type": "string", "enum": ["Y", "Z"], "default": "Y"},
                    "requireWatertight": {"type": "boolean", "default": False},
                    "maxRunSeconds": {"type": "integer", "minimum": 60, "maximum": 3600, "default": 1800},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        continuations=({
            "capability": "asset.3d.prepare.blender",
            "tool": "blender.prepare_game_asset",
            "outputs": [
                {"role": "game_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "preview", "kind": "image", "mediaTypes": ["image/png"], "required": True},
                {"role": "blender_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ],
            "constraintsSchema": {
                "type": "object",
                "properties": {
                    "targetFaces": {"type": "integer", "minimum": 100, "maximum": 1000000},
                    "targetEngine": {"type": "string", "enum": ["generic", "blender", "godot", "unity", "unreal"]},
                },
                "required": ["targetFaces", "targetEngine"],
                "additionalProperties": False,
            },
        },),
    )
    _SAFE_CONTAINER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    _OWNER_LABEL = "go7.spark-broker.owner"
    _MEDIA_EXTENSION = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }

    def __init__(self, workload_root: Path, *, broker_id: str, stop_containers: tuple[str, ...] = ()) -> None:
        self.workload_root = workload_root.resolve()
        if any(not self._SAFE_CONTAINER.fullmatch(name) for name in stop_containers):
            raise ValueError("stop container names must be literal Docker names")
        if stop_containers:
            raise ValueError("cross-profile SPARK_STOP_CONTAINERS is unsafe; configure fenced resource controllers instead")
        self.broker_id = broker_id
        self.runner = CommandRunner()

    def available(self) -> bool:
        return (self.workload_root / "compose.yaml").is_file() and (self.workload_root / "assets").is_dir() and (self.workload_root / "out").is_dir()

    def has_managed_unload(self) -> bool:
        # The current adapter launches a job-scoped compose container inside
        # execute(), after the coordinator's activation check. Cleanup is
        # broker-owned, but the runtime is not yet probe-observable at the
        # activation barrier, so it must not participate in profile rotation.
        return False

    def deactivate_plan(self, plan: ExecutionPlan) -> bool:
        del plan
        return True

    def activate(self, cancelled: Callable[[], bool]) -> None:
        self._cleanup_stale_job_containers(cancelled)

    def startup_cleanup(self) -> None:
        self._cleanup_stale_job_containers(lambda: False)

    def _cleanup_stale_job_containers(self, cancelled: Callable[[], bool]) -> None:
        if cancelled():
            raise ExecutionCancelled("job was cancelled")
        try:
            listed = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"label={self._OWNER_LABEL}={self.broker_id}"],
                cwd=self.workload_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionFailure("docker_timeout", "timed out inspecting stale Spark job containers", retryable=True) from exc
        if listed.returncode != 0:
            raise ExecutionFailure("docker_unavailable", "could not inspect stale Spark job containers", retryable=True)
        for container_id in listed.stdout.splitlines():
            container_id = container_id.strip()
            if not re.fullmatch(r"[a-f0-9]{12,64}", container_id):
                continue
            if cancelled():
                raise ExecutionCancelled("job was cancelled")
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    cwd=self.workload_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ExecutionFailure("docker_timeout", "timed out removing a stale Spark job container", retryable=True) from exc

    def validate_request(self, job: dict[str, Any], registry: ArtifactRegistry) -> None:
        request = job["request"]
        constraints = self._constraints(request.get("constraints", {}))
        if len(request["inputs"]) != 1:
            raise ContractError("invalid_inputs", "asset.3d.generate requires exactly one input image", field="inputs")
        input_artifact, _ = registry.resolve(request["inputs"][0]["artifactId"], verify=True)
        if input_artifact["kind"] != "image" or input_artifact["mediaType"] not in self._MEDIA_EXTENSION:
            raise ContractError("invalid_input", "asset.3d.generate requires a PNG, JPEG, or WEBP image artifact", field="inputs[0]")
        expected_sha = request["inputs"][0].get("sha256")
        if expected_sha and expected_sha != input_artifact["sha256"]:
            raise ContractError("artifact_hash_mismatch", "input artifact hash does not match request", field="inputs[0].sha256")
        self._validate_outputs(request.get("requiredOutputs", []), constraints["mode"])

    def execute(
        self,
        job: dict[str, Any],
        registry: ArtifactRegistry,
        cancelled: Callable[[], bool],
        stage: Callable[[str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        if not self.available():
            raise ExecutionFailure("profile_unavailable", "Hunyuan3D workload is not installed", retryable=True)
        request = job["request"]
        constraints = self._constraints(request.get("constraints", {}))
        workflow = request["workflow"]
        blender_preparation_authorized = (
            "asset.3d.prepare.blender" in workflow["approvedCapabilities"]
            and workflow["maxContinuations"] > 0
        )
        if len(request["inputs"]) != 1:
            raise ContractError("invalid_inputs", "asset.3d.generate requires exactly one input image", field="inputs")
        input_artifact, input_path = registry.resolve(request["inputs"][0]["artifactId"], verify=True)
        if input_artifact["kind"] != "image" or input_artifact["mediaType"] not in self._MEDIA_EXTENSION:
            raise ContractError("invalid_input", "asset.3d.generate requires a PNG, JPEG, or WEBP image artifact", field="inputs[0]")
        expected_sha = request["inputs"][0].get("sha256")
        if expected_sha and expected_sha != input_artifact["sha256"]:
            raise ContractError("artifact_hash_mismatch", "input artifact hash does not match request", field="inputs[0].sha256")
        self._validate_outputs(request.get("requiredOutputs", []), constraints["mode"])

        suffix = job["id"].removeprefix("job_")
        extension = self._MEDIA_EXTENSION[input_artifact["mediaType"]]
        source_name = f"broker-{suffix}{extension}"
        shape_name = f"broker-{suffix}-shape.glb"
        pbr_name = f"broker-{suffix}-pbr.glb"
        shape_report_name = f"broker-{suffix}-shape.json"
        pbr_report_name = f"broker-{suffix}-pbr.json"
        asset_target = self.workload_root / "assets" / source_name
        output_paths = [
            self.workload_root / "out" / shape_name,
            self.workload_root / "out" / pbr_name,
            self.workload_root / "out" / shape_report_name,
            self.workload_root / "out" / pbr_report_name,
        ]
        log_path = registry.root / ".staging" / f"{job['id']}.log"
        container_names: list[str] = []

        def remove_container(name: str) -> None:
            # Cleanup runs while reporting cancellation/failure and must not
            # mask that primary outcome if Docker is already unhealthy.
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name], cwd=self.workload_root,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=30,
                )
            except subprocess.TimeoutExpired:
                pass

        def run_compose(stage_name: str, service: str, args: list[str], timeout: int) -> None:
            container_name = f"spark-{suffix[:20]}-{stage_name}"
            container_names.append(container_name)
            self.runner.run(
                [
                    "docker", "compose", "run", "--rm", "--name", container_name,
                    "--label", f"{self._OWNER_LABEL}={self.broker_id}",
                    "--label", f"go7.spark-broker.job={job['id']}", service, *args,
                ],
                cwd=self.workload_root,
                log_path=log_path,
                timeout=timeout,
                cancelled=cancelled,
                cleanup=lambda: remove_container(container_name),
            )

        artifacts: list[dict[str, Any]] = []
        try:
            shutil.copyfile(input_path, asset_target)
            stage("running", {"stage": "shape_generation", "profileId": self.capability.profile_id})
            run_compose(
                "shape",
                "shape",
                ["scripts/shape_infer.py", "--image", f"/workspace/assets/{source_name}", "--out", f"/workspace/out/{shape_name}", "--seed", str(constraints["seed"])],
                constraints["maxRunSeconds"],
            )
            shape_path = self.workload_root / "out" / shape_name
            shape_generated_sha = self._file_sha256(shape_path)
            stage("validating", {"stage": "shape_validation"})
            run_compose(
                "validate-shape",
                "shape",
                ["scripts/validate_mesh.py", f"/workspace/out/{shape_name}", "--json", f"/workspace/out/{shape_report_name}"],
                300,
            )
            shape_validated_sha = self._file_sha256(shape_path)
            if shape_validated_sha != shape_generated_sha:
                raise ExecutionFailure("mesh_changed_during_validation", "shape mesh changed while it was being validated")
            shape_report = self._report(
                self.workload_root / "out" / shape_report_name,
                constraints,
                allow_mesh_repair=blender_preparation_authorized,
            )
            self._persist_report(self.workload_root / "out" / shape_report_name, shape_report)
            shape_artifact = registry.import_file(
                shape_path,
                kind="model3d",
                role="shape_model",
                media_type="model/gltf-binary",
                job_id=job["id"],
                metadata={"format": "glb", "units": constraints["units"], "upAxis": constraints["upAxis"], "sourceArtifactId": input_artifact["id"]},
                validation=shape_report,
                expected_sha256=shape_validated_sha,
            )
            artifacts.append(shape_artifact)
            report_artifact = registry.import_file(
                self.workload_root / "out" / shape_report_name,
                kind="report",
                role="mesh_report",
                media_type="application/json",
                job_id=job["id"],
                metadata={"modelArtifactId": shape_artifact["id"]},
                validation={"schema": "go7.mesh-validation.v1", "valid": bool(shape_report.get("valid"))},
                expected_sha256=self._file_sha256(self.workload_root / "out" / shape_report_name),
            )
            artifacts.append(report_artifact)

            final_model = shape_artifact
            if constraints["mode"] == "pbr":
                stage("running", {"stage": "pbr_texture"})
                run_compose(
                    "pbr",
                    "full",
                    [
                        "scripts/texture_infer.py", "--mesh", f"/workspace/out/{shape_name}",
                        "--image", f"/workspace/assets/{source_name}", "--out", f"/workspace/out/{pbr_name}",
                        "--max-views", str(constraints["textureViews"]), "--resolution", str(constraints["textureResolution"]),
                        "--seed", str(constraints["seed"]),
                    ],
                    constraints["maxRunSeconds"],
                )
                pbr_path = self.workload_root / "out" / pbr_name
                pbr_generated_sha = self._file_sha256(pbr_path)
                stage("validating", {"stage": "pbr_validation"})
                run_compose(
                    "validate-pbr",
                    "shape",
                    ["scripts/validate_mesh.py", f"/workspace/out/{pbr_name}", "--json", f"/workspace/out/{pbr_report_name}"],
                    300,
                )
                pbr_validated_sha = self._file_sha256(pbr_path)
                if pbr_validated_sha != pbr_generated_sha:
                    raise ExecutionFailure("mesh_changed_during_validation", "PBR mesh changed while it was being validated")
                pbr_report = self._report(
                    self.workload_root / "out" / pbr_report_name,
                    constraints,
                    allow_mesh_repair=blender_preparation_authorized,
                )
                self._persist_report(self.workload_root / "out" / pbr_report_name, pbr_report)
                final_model = registry.import_file(
                    pbr_path,
                    kind="model3d",
                    role="pbr_model",
                    media_type="model/gltf-binary",
                    job_id=job["id"],
                    metadata={"format": "glb", "units": constraints["units"], "upAxis": constraints["upAxis"], "sourceArtifactId": input_artifact["id"], "derivedFrom": shape_artifact["id"]},
                    validation=pbr_report,
                    expected_sha256=pbr_validated_sha,
                )
                artifacts.append(final_model)
                artifacts.append(registry.import_file(
                    self.workload_root / "out" / pbr_report_name,
                    kind="report",
                    role="pbr_mesh_report",
                    media_type="application/json",
                    job_id=job["id"],
                    metadata={"modelArtifactId": final_model["id"]},
                    validation={"schema": "go7.mesh-validation.v1", "valid": bool(pbr_report.get("valid"))},
                    expected_sha256=self._file_sha256(self.workload_root / "out" / pbr_report_name),
                ))
            if log_path.exists():
                artifacts.append(registry.import_file(
                    log_path,
                    kind="log",
                    role="execution_log",
                    media_type="text/plain",
                    job_id=job["id"],
                    metadata={"executor": "hunyuan3d"},
                ))
            continuation = self._blender_continuation(job, final_model)
            return {
                "artifacts": artifacts,
                "continuations": [continuation],
                "data": {
                    "primaryArtifactId": final_model["id"],
                    "mode": constraints["mode"],
                    "requiresPreparation": bool(final_model["validation"].get("requiresPreparation")),
                },
            }
        finally:
            asset_target.unlink(missing_ok=True)
            for path in output_paths:
                path.unlink(missing_ok=True)
            log_path.unlink(missing_ok=True)
            for container_name in container_names:
                remove_container(container_name)

    @staticmethod
    def _constraints(value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"mode", "seed", "textureViews", "textureResolution", "maxFaces", "targetEngine", "units", "upAxis", "requireWatertight", "maxRunSeconds"}
        unknown = set(value) - allowed
        if unknown:
            raise ContractError("unknown_constraint", f"asset.3d.generate has unknown constraints: {sorted(unknown)}", field="constraints")
        result = {
            "mode": value.get("mode", "shape"),
            "seed": value.get("seed", 42),
            "textureViews": value.get("textureViews", 6),
            "textureResolution": value.get("textureResolution", 512),
            "maxFaces": value.get("maxFaces", 1_000_000),
            "targetEngine": value.get("targetEngine", "generic"),
            "units": value.get("units", "meters"),
            "upAxis": value.get("upAxis", "Y"),
            "requireWatertight": value.get("requireWatertight", False),
            "maxRunSeconds": value.get("maxRunSeconds", 1800),
        }
        if result["mode"] not in {"shape", "pbr"}:
            raise ContractError("invalid_constraint", "constraints.mode must be shape or pbr", field="constraints.mode")
        for key, low, high in (("seed", 0, 2**31 - 1), ("textureViews", 1, 16), ("textureResolution", 128, 2048), ("maxFaces", 100, 10_000_000), ("maxRunSeconds", 60, 3600)):
            if not isinstance(result[key], int) or isinstance(result[key], bool) or not low <= result[key] <= high:
                raise ContractError("invalid_constraint", f"constraints.{key} must be an integer from {low} through {high}", field=f"constraints.{key}")
        if result["targetEngine"] not in {"generic", "blender", "godot", "unity", "unreal"}:
            raise ContractError("invalid_constraint", "unsupported targetEngine", field="constraints.targetEngine")
        if result["units"] not in {"meters", "centimeters"} or result["upAxis"] not in {"Y", "Z"}:
            raise ContractError("invalid_constraint", "unsupported units or upAxis", field="constraints")
        if not isinstance(result["requireWatertight"], bool):
            raise ContractError("invalid_constraint", "requireWatertight must be boolean", field="constraints.requireWatertight")
        return result

    @staticmethod
    def _validate_outputs(outputs: list[dict[str, Any]], mode: str) -> None:
        supported = {
            "shape_model": ("model3d", {"model/gltf-binary"}),
            "pbr_model": ("model3d", {"model/gltf-binary"}),
            "mesh_report": ("report", {"application/json"}),
            "pbr_mesh_report": ("report", {"application/json"}),
            "execution_log": ("log", {"text/plain"}),
        }
        for output in outputs:
            spec = supported.get(output["role"])
            if spec is None:
                if output["required"]:
                    raise ContractError("unsupported_output", f"required output role {output['role']} is not supported", field="requiredOutputs")
                continue
            kind, media_types = spec
            if output["kind"] != kind or not media_types.intersection(output["mediaTypes"]):
                raise ContractError("unsupported_media_type", f"no supported media type for {output['role']}", field="requiredOutputs")
            if output["role"].startswith("pbr") and mode != "pbr" and output["required"]:
                raise ContractError("unsupported_output", f"{output['role']} requires constraints.mode=pbr", field="requiredOutputs")

    @staticmethod
    def _report(path: Path, constraints: dict[str, Any], *, allow_mesh_repair: bool = False) -> dict[str, Any]:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionFailure("invalid_validation_report", "mesh validator did not produce valid JSON") from exc
        if not report.get("valid"):
            raise ExecutionFailure("mesh_invalid", "generated mesh failed validation")
        face_limit_satisfied = int(report.get("faces", 0)) <= constraints["maxFaces"]
        if not face_limit_satisfied and not allow_mesh_repair:
            raise ExecutionFailure("face_limit_exceeded", f"generated mesh exceeds {constraints['maxFaces']} faces")
        watertight_satisfied = int(report.get("watertight_geometry", 0)) >= int(report.get("geometry_count", 1))
        if constraints["requireWatertight"] and not watertight_satisfied and not allow_mesh_repair:
            raise ExecutionFailure("not_watertight", "generated mesh is not fully watertight")
        report["faceLimitSatisfied"] = face_limit_satisfied
        report["watertightSatisfied"] = watertight_satisfied
        report["requiresPreparation"] = not face_limit_satisfied or (constraints["requireWatertight"] and not watertight_satisfied)
        return report

    @staticmethod
    def _persist_report(path: Path, report: dict[str, Any]) -> None:
        staged = path.with_name(f".{path.name}.validated-{os.getpid()}-{threading.get_ident()}")
        try:
            with staged.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ExecutionFailure("output_unavailable", "generated output cannot be opened safely") from exc
        digest = hashlib.sha256()
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExecutionFailure("output_unavailable", "generated output is not a regular file")
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        return digest.hexdigest()

    @staticmethod
    def _blender_continuation(job: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
        workflow = job["request"]["workflow"]
        capability = "asset.3d.prepare.blender"
        approved = capability in workflow["approvedCapabilities"]
        eligible = bool(workflow["autoContinue"] and approved and workflow["maxContinuations"] > 0)
        return {
            "id": f"cont_{job['id'].removeprefix('job_')}",
            "capability": capability,
            "tool": "blender.prepare_game_asset",
            "inputBindings": [{"name": "sourceModel", "artifactId": model["id"], "sha256": model["sha256"], "mediaType": model["mediaType"]}],
            "requiredOutputs": [
                {"role": "game_model", "kind": "model3d", "mediaTypes": ["model/gltf-binary"], "required": True},
                {"role": "preview", "kind": "image", "mediaTypes": ["image/png"], "required": True},
                {"role": "blender_report", "kind": "report", "mediaTypes": ["application/json"], "required": True},
            ],
            "constraints": {"targetFaces": min(100_000, int(job["request"]["constraints"].get("maxFaces", 100_000))), "targetEngine": job["request"]["constraints"].get("targetEngine", "generic")},
            "authorization": {"mode": "explicit", "approvedByRequest": approved},
            "autoStartEligible": eligible,
            "idempotencyKey": f"{job['id']}:blender:v1",
        }


class OpenAIChatExecutor(Executor):
    _invocation = {
        "inputs": [{
            "role": "prompt", "kind": "text",
            "mediaTypes": ["text/plain", "application/json"],
            "required": True, "minItems": 1, "maxItems": 1,
        }],
        "outputs": [
            {"role": "text_output", "kind": "text", "mediaTypes": ["text/plain"], "required": True},
            {"role": "provider_response", "kind": "report", "mediaTypes": ["application/json"], "required": True},
        ],
        "constraintsSchema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.2},
                "maxTokens": {"type": "integer", "minimum": 1, "maximum": 32768, "default": 1024},
                "systemPrompt": {"type": "string", "maxLength": 16384, "default": ""},
                "timeoutSeconds": {"type": "integer", "minimum": 10, "maximum": 1800, "default": 600},
                "enableThinking": {"type": "boolean", "default": False},
            },
            "required": [],
            "additionalProperties": False,
        },
    }

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        container_name: str | None,
        profile_id: str = "gpu.openai-compatible",
        description: str = "Generate text with the configured local OpenAI-compatible model",
        estimated_memory_gb: int = 0,
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9]*(?:\.[a-z0-9][a-z0-9-]*){1,7}", profile_id):
            raise ValueError("text profile id is invalid")
        if not description.strip() or len(description) > 256:
            raise ValueError("text profile description is invalid")
        if not isinstance(estimated_memory_gb, int) or isinstance(estimated_memory_gb, bool) or not 0 <= estimated_memory_gb <= 1024:
            raise ValueError("text profile estimated memory must be 0-1024 GiB")
        self.capability = Capability(
            id="text.chat.generate",
            profile_id=profile_id,
            description=description.strip(),
            input_kinds=("text",),
            output_roles=("text_output", "provider_response"),
            estimated_memory_gb=estimated_memory_gb,
            invocation=self._invocation,
            service_class="interactive",
            lease_mode="permit",
        )
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("local text endpoint must be an HTTP loopback URL")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.container_name = container_name
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        if container_name and not Hunyuan3DExecutor._SAFE_CONTAINER.fullmatch(container_name):
            raise ValueError("text container must be a literal Docker name")

    def activate(self, cancelled: Callable[[], bool]) -> None:
        if self.container_name:
            try:
                inspected = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.Running}}", self.container_name],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                    text=True,
                )
                already_running = inspected.returncode == 0 and inspected.stdout.strip().lower() == "true"
                if not already_running:
                    result = subprocess.run(
                        ["docker", "start", self.container_name],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=60,
                        check=False,
                    )
                    if result.returncode != 0:
                        raise ExecutionFailure("profile_load_failed", f"could not start text profile container {self.container_name}", retryable=True)
            except subprocess.TimeoutExpired as exc:
                raise ExecutionFailure("profile_load_timeout", f"timed out loading text profile container {self.container_name}", retryable=True) from exc
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if cancelled():
                raise ExecutionCancelled("job was cancelled while loading text profile")
            try:
                request = urllib.request.Request(f"{self.endpoint}/readyz", headers={"Authorization": f"Bearer {self.api_key}"})
                with self._opener.open(request, timeout=10) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise ExecutionFailure("profile_load_timeout", "text model did not become ready within 900 seconds", retryable=True)

    def plan(self, job: dict[str, Any], routing_context: dict[str, Any] | None = None) -> ExecutionPlan:
        base = super().plan(job, routing_context)
        return ExecutionPlan(
            profile_id=base.profile_id,
            route_id=base.route_id,
            resource_group=base.resource_group,
            service_class=base.service_class,
            lease_mode=base.lease_mode,
            estimated_memory_gb=base.estimated_memory_gb,
            preemption_mode=base.preemption_mode,
            route_reason=base.route_reason,
            verify_profile_active=True,
        )

    def deactivate_plan(self, plan: ExecutionPlan) -> bool:
        del plan
        if not self.container_name:
            return False
        try:
            result = subprocess.run(
                ["docker", "stop", "--time", "30", self.container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionFailure(
                "profile_unload_timeout", f"timed out unloading text profile container {self.container_name}",
                retryable=False,
            ) from exc
        if result.returncode != 0:
            raise ExecutionFailure(
                "profile_unload_failed", f"could not stop text profile container {self.container_name}",
                retryable=False,
            )
        return True

    def has_managed_unload(self) -> bool:
        return self.container_name is not None

    def validate_request(self, job: dict[str, Any], registry: ArtifactRegistry) -> None:
        request = job["request"]
        if len(request["inputs"]) != 1:
            raise ContractError("invalid_inputs", "text.chat.generate requires one text artifact", field="inputs")
        artifact, path = registry.resolve(request["inputs"][0]["artifactId"], verify=True)
        if artifact["kind"] != "text" or artifact["mediaType"] not in {"text/plain", "application/json"}:
            raise ContractError("invalid_input", "text.chat.generate requires a text/plain or application/json text artifact", field="inputs[0]")
        expected_sha = request["inputs"][0].get("sha256")
        if expected_sha and expected_sha != artifact["sha256"]:
            raise ContractError("artifact_hash_mismatch", "input artifact hash does not match request", field="inputs[0].sha256")
        self._constraints(request.get("constraints", {}))
        self._validate_outputs(request.get("requiredOutputs", []))
        if path.stat().st_size > 1024 * 1024:
            raise ContractError("input_too_large", "text input exceeds 1 MiB", field="inputs[0]")

    def execute(
        self,
        job: dict[str, Any],
        registry: ArtifactRegistry,
        cancelled: Callable[[], bool],
        stage: Callable[[str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        request = job["request"]
        if len(request["inputs"]) != 1:
            raise ContractError("invalid_inputs", "text.chat.generate requires one text artifact", field="inputs")
        artifact, path = registry.resolve(request["inputs"][0]["artifactId"], verify=True)
        if artifact["kind"] != "text" or artifact["mediaType"] not in {"text/plain", "application/json"}:
            raise ContractError("invalid_input", "text.chat.generate requires a text/plain or application/json text artifact", field="inputs[0]")
        expected_sha = request["inputs"][0].get("sha256")
        if expected_sha and expected_sha != artifact["sha256"]:
            raise ContractError("artifact_hash_mismatch", "input artifact hash does not match request", field="inputs[0].sha256")
        constraints = self._constraints(request.get("constraints", {}))
        self._validate_outputs(request.get("requiredOutputs", []))
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ContractError("input_too_large", "text input exceeds 1 MiB", field="inputs[0]")
        try:
            if artifact["mediaType"] == "application/json":
                conversation = json.loads(raw)
                if not isinstance(conversation, list) or not conversation:
                    raise ValueError
                messages = []
                for item in conversation:
                    if not isinstance(item, dict) or set(item) != {"role", "content"} or item["role"] not in {"system", "user", "assistant"} or not isinstance(item["content"], str):
                        raise ValueError
                    messages.append({"role": item["role"], "content": item["content"]})
            else:
                prompt = raw.decode("utf-8")
                messages = []
                if constraints["systemPrompt"]:
                    messages.append({"role": "system", "content": constraints["systemPrompt"]})
                messages.append({"role": "user", "content": prompt})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ContractError("invalid_text_input", "text artifact is not a valid UTF-8 prompt or conversation", field="inputs[0]") from exc
        if cancelled():
            raise ExecutionCancelled("job was cancelled")
        stage("running", {"stage": "text_generation", "model": self.model})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": constraints["temperature"],
            "max_tokens": constraints["maxTokens"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": constraints["enableThinking"]},
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=encoded,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        request_timeout = constraints["timeoutSeconds"]
        bounded_timeout = getattr(cancelled, "bounded_timeout", None)
        if callable(bounded_timeout):
            request_timeout = bounded_timeout(request_timeout)
        try:
            with self._opener.open(http_request, timeout=request_timeout) as response:
                limit = 4 * 1024 * 1024
                response_bytes = response.read(limit + 1)
                if len(response_bytes) > limit:
                    raise ExecutionFailure("provider_response_too_large", "text provider response exceeded 4 MiB")
        except urllib.error.HTTPError as exc:
            message = f"text provider returned HTTP {exc.code}"
            exc.close()
            raise ExecutionFailure("provider_error", message, retryable=exc.code >= 500) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ExecutionFailure("provider_unavailable", "text provider was unavailable", retryable=True) from exc
        if cancelled():
            raise ExecutionCancelled("job was cancelled after provider response")
        stage("validating", {"stage": "text_response_validation"})
        try:
            response_json = json.loads(response_bytes)
            choice = response_json["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise TypeError
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ExecutionFailure("invalid_provider_response", "text provider response did not match the OpenAI chat schema") from exc
        if response_json.get("model") != self.model:
            raise ExecutionFailure(
                "provider_model_mismatch",
                "text provider response model did not match the selected installed route",
                retryable=False,
            )
        work = registry.root / ".staging"
        text_path = work / f"{job['id']}-text.txt"
        json_path = work / f"{job['id']}-provider.json"
        try:
            text_path.write_text(content, encoding="utf-8")
            json_path.write_bytes(response_bytes)
            usage = response_json.get("usage", {}) if isinstance(response_json.get("usage"), dict) else {}
            text_artifact = registry.import_file(
                text_path, kind="text", role="text_output", media_type="text/plain", job_id=job["id"],
                metadata={"model": self.model, "sourceArtifactId": artifact["id"]},
                validation={"schema": "openai.chat.completion.v1", "finishReason": choice.get("finish_reason"), "usage": usage, "valid": True},
            )
            response_artifact = registry.import_file(
                json_path, kind="report", role="provider_response", media_type="application/json", job_id=job["id"],
                metadata={"model": self.model, "textArtifactId": text_artifact["id"]},
                validation={"schema": "openai.chat.completion.v1", "valid": True},
            )
            return {"artifacts": [text_artifact, response_artifact], "continuations": [], "data": {"primaryArtifactId": text_artifact["id"], "model": self.model, "usage": usage}}
        finally:
            text_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)

    @staticmethod
    def _constraints(value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"temperature", "maxTokens", "systemPrompt", "timeoutSeconds", "enableThinking"}
        unknown = set(value) - allowed
        if unknown:
            raise ContractError("unknown_constraint", f"text.chat.generate has unknown constraints: {sorted(unknown)}", field="constraints")
        temperature = value.get("temperature", 0.2)
        max_tokens = value.get("maxTokens", 1024)
        system_prompt = value.get("systemPrompt", "")
        timeout = value.get("timeoutSeconds", 600)
        enable_thinking = value.get("enableThinking", False)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
            raise ContractError("invalid_constraint", "temperature must be 0-2", field="constraints.temperature")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 32768:
            raise ContractError("invalid_constraint", "maxTokens must be 1-32768", field="constraints.maxTokens")
        if not isinstance(system_prompt, str) or len(system_prompt) > 16384:
            raise ContractError("invalid_constraint", "systemPrompt must be at most 16384 characters", field="constraints.systemPrompt")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 10 <= timeout <= 1800:
            raise ContractError("invalid_constraint", "timeoutSeconds must be 10-1800", field="constraints.timeoutSeconds")
        if not isinstance(enable_thinking, bool):
            raise ContractError("invalid_constraint", "enableThinking must be boolean", field="constraints.enableThinking")
        return {"temperature": float(temperature), "maxTokens": max_tokens, "systemPrompt": system_prompt, "timeoutSeconds": timeout, "enableThinking": enable_thinking}

    @staticmethod
    def _validate_outputs(outputs: list[dict[str, Any]]) -> None:
        supported = {
            "text_output": ("text", {"text/plain"}),
            "provider_response": ("report", {"application/json"}),
        }
        for output in outputs:
            spec = supported.get(output["role"])
            if spec is None:
                if output["required"]:
                    raise ContractError("unsupported_output", f"required output role {output['role']} is not supported", field="requiredOutputs")
                continue
            kind, media_types = spec
            if output["kind"] != kind or not media_types.intersection(output["mediaTypes"]):
                raise ContractError("unsupported_media_type", f"no supported output definition for {output['role']}", field="requiredOutputs")


@dataclass(frozen=True)
class OpenAIRoute:
    id: str
    model: str
    profile_id: str
    description: str
    endpoint: str
    api_key: str
    container_name: str | None
    estimated_memory_gb: int
    priority: int
    service_classes: tuple[str, ...]
    config_revision: str | None = None

    def public(self) -> dict[str, Any]:
        value = {
            "id": self.id,
            "model": self.model,
            "profileId": self.profile_id,
            "description": self.description,
            "estimatedMemoryGb": self.estimated_memory_gb,
            "priority": self.priority,
            "serviceClasses": list(self.service_classes),
        }
        if self.config_revision is not None:
            value["configRevision"] = self.config_revision
        return value


def load_openai_routes(path: Path) -> tuple[OpenAIRoute, ...]:
    try:
        value = json.loads(read_owner_text(path, "OpenAI routes", maximum_bytes=1024 * 1024))
    except SecureFileError as exc:
        raise ValueError(str(exc)) from exc
    try:
        compiled = compile_routing_config(value)
    except RoutingError as exc:
        raise ValueError(str(exc)) from exc
    raw_by_id = {item["id"]: item for item in value["routes"]}
    result: list[OpenAIRoute] = []
    for route in compiled.routes:
        item = raw_by_id[route.id]
        key_file = item["apiKeyFile"]
        key_path = Path(key_file)
        try:
            api_key = read_owner_secret(
                key_path,
                "OpenAI route credential",
                minimum_length=8,
            )
        except SecureFileError as exc:
            raise ValueError(str(exc)) from exc
        result.append(OpenAIRoute(
            id=route.id,
            model=route.model,
            profile_id=route.profile_id,
            description=item.get("description", f"Local model route {route.id}").strip(),
            endpoint=item["endpoint"].rstrip("/"),
            api_key=api_key,
            container_name=item.get("container"),
            estimated_memory_gb=route.estimated_memory_gb,
            priority=route.priority,
            service_classes=route.service_classes,
            config_revision=compiled.revision,
        ))
    return tuple(result)


class RoutedOpenAIChatExecutor(Executor):
    def __init__(self, routes: tuple[OpenAIRoute, ...]) -> None:
        if not routes:
            raise ValueError("at least one inference route is required")
        self.routes = routes
        declared_revisions = {route.config_revision for route in routes if route.config_revision is not None}
        if declared_revisions and (
            len(declared_revisions) != 1 or any(route.config_revision is None for route in routes)
        ):
            raise ValueError("inference routes must share one compiled configuration revision")
        self.installed_config_revision = next(iter(declared_revisions), None)
        try:
            self.routing_config = compile_route_policies(
                ({
                    "id": route.id,
                    "model": route.model,
                    "profileId": route.profile_id,
                    "estimatedMemoryGb": route.estimated_memory_gb,
                    "priority": route.priority,
                    "serviceClasses": list(route.service_classes),
                } for route in routes),
            )
        except RoutingError as exc:
            raise ValueError(str(exc)) from exc
        self.routing_engine = self.routing_config.engine()
        self.backends = {
            route.id: OpenAIChatExecutor(
                endpoint=route.endpoint,
                api_key=route.api_key,
                model=route.model,
                container_name=route.container_name,
                profile_id=route.profile_id,
                description=route.description,
                estimated_memory_gb=route.estimated_memory_gb,
            )
            for route in routes
        }
        invocation = json.loads(json.dumps(OpenAIChatExecutor._invocation))
        properties = invocation["constraintsSchema"]["properties"]
        properties.update({
            "model": {"type": "string", "enum": sorted({route.model for route in routes})},
            "serviceClass": {"type": "string", "enum": ["interactive", "batch", "background"], "default": "interactive"},
            "routePreference": {"type": "string", "enum": ["balanced", "latency", "throughput", "memory"], "default": "balanced"},
        })
        self.capability = Capability(
            id="text.chat.generate",
            profile_id="gpu.routed-inference",
            description="Route text generation across administrator-installed local model profiles",
            input_kinds=("text",),
            output_roles=("text_output", "provider_response"),
            estimated_memory_gb=max(route.estimated_memory_gb for route in routes),
            invocation=invocation,
            service_class="interactive",
            lease_mode="permit",
        )

    def public_routes(self) -> list[dict[str, Any]]:
        return [route.public() for route in sorted(self.routes, key=lambda item: item.id)]

    def has_managed_unload(self) -> bool:
        return all(backend.has_managed_unload() for backend in self.backends.values())

    def gpu_profile_lifecycles(self) -> dict[str, bool]:
        return {
            route.profile_id: self.backends[route.id].has_managed_unload()
            for route in self.routes
        }

    def validate_request(self, job: dict[str, Any], registry: ArtifactRegistry) -> None:
        plan = self.plan(job)
        self.backends[plan.route_id or ""].validate_request(self._delegate_job(job), registry)

    def plan(self, job: dict[str, Any], routing_context: dict[str, Any] | None = None) -> ExecutionPlan:
        constraints = job["request"].get("constraints", {})
        allowed = {"temperature", "maxTokens", "systemPrompt", "timeoutSeconds", "enableThinking", "model", "serviceClass", "routePreference"}
        unknown = set(constraints) - allowed
        if unknown:
            raise ContractError("unknown_constraint", f"text.chat.generate has unknown constraints: {sorted(unknown)}", field="constraints")
        model = constraints.get("model")
        if model is not None and (not isinstance(model, str) or model not in {route.model for route in self.routes}):
            raise ContractError("route_unavailable", "the requested model is not installed", field="constraints.model")
        service_class = constraints.get("serviceClass", "interactive")
        if service_class not in {"interactive", "batch", "background"}:
            raise ContractError("invalid_constraint", "serviceClass is invalid", field="constraints.serviceClass")
        preference = constraints.get("routePreference", "balanced")
        if preference not in {"balanced", "latency", "throughput", "memory"}:
            raise ContractError("invalid_constraint", "routePreference is invalid", field="constraints.routePreference")
        try:
            decision = self.routing_engine.decide(
                model=model,
                service_class=service_class,
                preference=preference,
                snapshot=routing_context,
            )
        except RoutingError as exc:
            field = "constraints"
            if exc.field == "model":
                field = "constraints.model"
            elif exc.field == "serviceClass":
                field = "constraints.serviceClass"
            elif exc.field == "routePreference":
                field = "constraints.routePreference"
            raise ContractError(exc.code, str(exc), field=field) from exc
        selected = next(route for route in self.routes if route.id == decision.route_id)
        return ExecutionPlan(
            profile_id=selected.profile_id,
            route_id=selected.id,
            resource_group="gpu:0",
            service_class=service_class,
            lease_mode="permit",
            estimated_memory_gb=selected.estimated_memory_gb,
            route_reason=decision.reason,
            verify_profile_active=True,
        )

    def activate_plan(self, plan: ExecutionPlan, cancelled: Callable[[], bool]) -> None:
        self.backends[plan.route_id or ""].activate(cancelled)

    def deactivate_plan(self, plan: ExecutionPlan) -> bool:
        return self.backends[plan.route_id or ""].deactivate_plan(plan)

    def execute_plan(
        self,
        plan: ExecutionPlan,
        job: dict[str, Any],
        registry: ArtifactRegistry,
        cancelled: Callable[[], bool],
        stage: Callable[[str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        route = next(item for item in self.routes if item.id == plan.route_id)
        payload = self.backends[route.id].execute(self._delegate_job(job), registry, cancelled, stage)
        payload.setdefault("data", {})["route"] = {
            "id": route.id,
            "model": route.model,
            "profileId": route.profile_id,
            "reason": plan.route_reason,
        }
        return payload

    @staticmethod
    def _delegate_job(job: dict[str, Any]) -> dict[str, Any]:
        value = dict(job)
        request = dict(job["request"])
        constraints = dict(request.get("constraints", {}))
        for key in ("model", "serviceClass", "routePreference"):
            constraints.pop(key, None)
        request["constraints"] = constraints
        value["request"] = request
        return value


def build_executors(
    *,
    broker_id: str,
    hunyuan_root: Path | None,
    stop_containers: tuple[str, ...],
    text_endpoint: str | None = None,
    text_api_key: str = "",
    text_model: str = "",
    text_container: str | None = None,
    text_profile_id: str = "gpu.openai-compatible",
    text_description: str = "Generate text with the configured local OpenAI-compatible model",
    text_estimated_memory_gb: int = 0,
    openai_routes: tuple[OpenAIRoute, ...] = (),
) -> dict[str, Executor]:
    values: list[Executor] = [EchoExecutor()]
    if hunyuan_root is not None:
        hunyuan = Hunyuan3DExecutor(hunyuan_root, broker_id=broker_id, stop_containers=stop_containers)
        if hunyuan.available():
            values.append(hunyuan)
    if openai_routes:
        values.append(RoutedOpenAIChatExecutor(openai_routes))
    elif text_endpoint and text_api_key and text_model:
        values.append(OpenAIChatExecutor(
            endpoint=text_endpoint,
            api_key=text_api_key,
            model=text_model,
            container_name=text_container,
            profile_id=text_profile_id,
            description=text_description,
            estimated_memory_gb=text_estimated_memory_gb,
        ))
    return {executor.capability.id: executor for executor in values}
