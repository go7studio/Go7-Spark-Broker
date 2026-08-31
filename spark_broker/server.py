from __future__ import annotations

import hmac
import json
import os
import re
import signal
import threading
import urllib.parse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import BROKER_VERSION, PROTOCOL_VERSION
from .artifacts import ArtifactError, ArtifactIntegrityError, ArtifactNotFound, ArtifactRegistry
from .contract import ContractError, validate_job_request, validate_upload_metadata
from .executors import build_executors, load_openai_routes
from .resources import ResourceCoordinator, ResourcePolicy
from .secure_files import SecureFileError, read_owner_secret
from .scheduler import Scheduler
from .store import IdempotencyConflict, QueueFull, Store


_JOB_ID = re.compile(r"^job_[a-f0-9]{32}$")
_ARTIFACT_ID = re.compile(r"^art_[a-f0-9]{32}$")
_BROKER_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+){0,7}$")


@dataclass(frozen=True)
class Config:
    broker_id: str
    bind: str
    port: int
    token: str
    data_root: Path
    hunyuan_root: Path | None
    stop_containers: tuple[str, ...]
    text_endpoint: str | None = None
    text_api_key: str = ""
    text_model: str = ""
    text_container: str | None = None
    text_profile_id: str = "gpu.openai-compatible"
    text_description: str = "Generate text with the configured local OpenAI-compatible model"
    text_estimated_memory_gb: int = 0
    max_json_bytes: int = 2 * 1024 * 1024
    max_artifact_bytes: int = 2 * 1024 * 1024 * 1024
    max_hops: int = 8
    openai_routes_file: Path | None = None
    resource_policy_file: Path | None = None
    max_pending_jobs: int = 1000
    max_storage_bytes: int = 100 * 1024 * 1024 * 1024
    request_timeout_seconds: int = 30
    max_concurrent_uploads: int = 2
    coordinator_lock_file: Path | None = None
    coordinator_epoch_file: Path | None = None
    enforce_host_lock_scope: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        token = os.environ.get("SPARK_BROKER_TOKEN", "")
        token_file = os.environ.get("SPARK_BROKER_TOKEN_FILE", "")
        allow_inline = os.environ.get("SPARK_ALLOW_INLINE_SECRETS") == "1"
        if token and not allow_inline:
            raise SystemExit("inline broker secrets are refused; use SPARK_BROKER_TOKEN_FILE")
        if not token and token_file:
            token = _read_credential(Path(token_file), "broker token", minimum_length=32)
        if len(token) < 32:
            raise SystemExit("SPARK_BROKER_TOKEN or SPARK_BROKER_TOKEN_FILE must contain at least 32 characters")
        hunyuan = os.environ.get("SPARK_HUNYUAN_ROOT", "")
        text_key = os.environ.get("SPARK_OPENAI_API_KEY", os.environ.get("SPARK_TEXT_API_KEY", ""))
        text_key_file = os.environ.get("SPARK_OPENAI_API_KEY_FILE", os.environ.get("SPARK_TEXT_API_KEY_FILE", ""))
        if text_key and not allow_inline:
            raise SystemExit("inline model-server secrets are refused; use SPARK_OPENAI_API_KEY_FILE")
        if not text_key and text_key_file:
            text_key = _read_credential(Path(text_key_file), "model-server credential", minimum_length=8)
        text_endpoint = os.environ.get("SPARK_OPENAI_ENDPOINT", os.environ.get("SPARK_TEXT_ENDPOINT", ""))
        text_model = os.environ.get("SPARK_OPENAI_MODEL", os.environ.get("SPARK_TEXT_MODEL", ""))
        if any((text_endpoint, text_key, text_model)) and not all((text_endpoint, text_key, text_model)):
            raise SystemExit("OpenAI-compatible text configuration requires endpoint, credential, and model together")
        bind = os.environ.get("SPARK_BROKER_BIND", "127.0.0.1")
        if bind not in {"127.0.0.1", "localhost", "::1"} and os.environ.get("SPARK_ALLOW_INSECURE_NONLOOPBACK") != "1":
            raise SystemExit("SPARK_BROKER_BIND must be loopback; publish it through an authenticated HTTPS reverse proxy")
        return cls(
            broker_id=os.environ.get("SPARK_BROKER_ID", "local-capability-host"),
            bind=bind,
            port=int(os.environ.get("SPARK_BROKER_PORT", "8790")),
            token=token,
            data_root=Path(os.environ.get("SPARK_BROKER_DATA", "/var/lib/go7-spark-broker")),
            hunyuan_root=Path(hunyuan) if hunyuan else None,
            stop_containers=tuple(filter(None, os.environ.get("SPARK_STOP_CONTAINERS", "").split(","))),
            text_endpoint=text_endpoint or None,
            text_api_key=text_key,
            text_model=text_model,
            text_container=os.environ.get("SPARK_OPENAI_CONTAINER", os.environ.get("SPARK_TEXT_CONTAINER", "")) or None,
            text_profile_id=os.environ.get("SPARK_OPENAI_PROFILE_ID", "gpu.openai-compatible"),
            text_description=os.environ.get("SPARK_OPENAI_DESCRIPTION", "Generate text with the configured local OpenAI-compatible model"),
            text_estimated_memory_gb=int(os.environ.get("SPARK_OPENAI_ESTIMATED_MEMORY_GB", "0")),
            max_json_bytes=int(os.environ.get("SPARK_MAX_JSON_BYTES", str(2 * 1024 * 1024))),
            max_artifact_bytes=int(os.environ.get("SPARK_MAX_ARTIFACT_BYTES", str(2 * 1024 * 1024 * 1024))),
            max_hops=int(os.environ.get("SPARK_MAX_HOPS", "8")),
            openai_routes_file=Path(os.environ["SPARK_OPENAI_ROUTES_FILE"]) if os.environ.get("SPARK_OPENAI_ROUTES_FILE") else None,
            resource_policy_file=Path(os.environ["SPARK_RESOURCE_POLICY_FILE"]) if os.environ.get("SPARK_RESOURCE_POLICY_FILE") else None,
            max_pending_jobs=int(os.environ.get("SPARK_MAX_PENDING_JOBS", "1000")),
            max_storage_bytes=int(os.environ.get("SPARK_MAX_STORAGE_BYTES", str(100 * 1024 * 1024 * 1024))),
            request_timeout_seconds=int(os.environ.get("SPARK_REQUEST_TIMEOUT_SECONDS", "30")),
            max_concurrent_uploads=int(os.environ.get("SPARK_MAX_CONCURRENT_UPLOADS", "2")),
            coordinator_lock_file=Path(os.environ["SPARK_COORDINATOR_LOCK_FILE"]) if os.environ.get("SPARK_COORDINATOR_LOCK_FILE") else None,
            coordinator_epoch_file=Path(os.environ["SPARK_COORDINATOR_EPOCH_FILE"]) if os.environ.get("SPARK_COORDINATOR_EPOCH_FILE") else None,
            enforce_host_lock_scope=True,
        )


def _read_credential(path: Path, label: str, *, minimum_length: int) -> str:
    try:
        return read_owner_secret(path, label, minimum_length=minimum_length)
    except SecureFileError as exc:
        raise SystemExit(str(exc)) from exc


class Broker:
    def __init__(self, config: Config) -> None:
        self.config = config
        if not _BROKER_ID.fullmatch(config.broker_id):
            raise ValueError("broker_id is invalid")
        config.data_root.mkdir(parents=True, exist_ok=True)
        self.store = Store(config.data_root / "broker.sqlite3")
        if (
            config.max_pending_jobs < 1
            or config.max_storage_bytes < config.max_artifact_bytes
            or not 5 <= config.request_timeout_seconds <= 600
            or not 1 <= config.max_concurrent_uploads <= 64
        ):
            raise ValueError("queue and storage limits are inconsistent")
        self.registry = ArtifactRegistry(
            config.data_root / "artifacts", self.store,
            max_upload_bytes=config.max_artifact_bytes,
            max_storage_bytes=config.max_storage_bytes,
        )
        self.registry.clean_staging()
        routes = load_openai_routes(config.openai_routes_file) if config.openai_routes_file else ()
        self.executors = build_executors(
            broker_id=config.broker_id,
            hunyuan_root=config.hunyuan_root,
            stop_containers=config.stop_containers,
            text_endpoint=config.text_endpoint,
            text_api_key=config.text_api_key,
            text_model=config.text_model,
            text_container=config.text_container,
            text_profile_id=config.text_profile_id,
            text_description=config.text_description,
            text_estimated_memory_gb=config.text_estimated_memory_gb,
            openai_routes=routes,
        )
        policy = ResourcePolicy.from_file(config.resource_policy_file) if config.resource_policy_file else ResourcePolicy()
        gpu_installed = any(executor.capability.resource_group is not None for executor in self.executors.values())
        if gpu_installed:
            if (
                config.coordinator_lock_file is None
                or not config.coordinator_lock_file.is_absolute()
                or config.coordinator_epoch_file is None
                or not config.coordinator_epoch_file.is_absolute()
            ):
                raise ValueError("GPU capabilities require absolute coordinator lock and durable epoch files")
            if config.enforce_host_lock_scope:
                resolved_lock = config.coordinator_lock_file.resolve()
                resolved_epoch = config.coordinator_epoch_file.resolve()
                unsafe_roots = [Path("/tmp"), Path("/var/tmp")]
                if os.environ.get("XDG_RUNTIME_DIR"):
                    unsafe_roots.append(Path(os.environ["XDG_RUNTIME_DIR"]).resolve())
                if any(
                    value == root or root in value.parents
                    for value in (resolved_lock, resolved_epoch)
                    for root in unsafe_roots
                ):
                    raise ValueError("GPU coordinator lock must use an administrator-provisioned host-wide path")
            if (
                config.resource_policy_file is None
                or not policy.require_probe
                or not policy.enforce_memory_admission
                or not policy.enforce_cuda_admission
                or not policy.probe_endpoint
                or not policy.probe_token_file
            ):
                raise ValueError(
                    "GPU capabilities require a resource policy with required probe, "
                    "host-memory admission, and CUDA admission"
                )
            profile_lifecycles: dict[str, bool] = {}
            for executor in self.executors.values():
                for profile_id, managed in executor.gpu_profile_lifecycles().items():
                    profile_lifecycles[profile_id] = profile_lifecycles.get(profile_id, True) and managed
            gpu_profiles = set(profile_lifecycles)
            unmanaged_profiles = {
                profile_id for profile_id, managed in profile_lifecycles.items() if not managed
            }
            if unmanaged_profiles and (policy.controllers or len(gpu_profiles) > 1):
                raise ValueError(
                    "controller-backed or multi-profile GPU routing requires a "
                    "broker-managed unload lifecycle for every GPU profile"
                )
        self.coordinator = ResourceCoordinator(
            store=self.store,
            data_root=config.data_root,
            policy=policy,
            lock_path=config.coordinator_lock_file,
            epoch_path=config.coordinator_epoch_file,
        )
        self.scheduler = Scheduler(
            broker_id=config.broker_id,
            store=self.store,
            registry=self.registry,
            executors=self.executors,
            coordinator=self.coordinator,
        )

    def start(self) -> None:
        self.coordinator.start()
        try:
            if self.coordinator.quarantined:
                return
            # Crash leftovers must be removed before the scheduler can load a
            # different GPU profile onto the same device. Quarantined state is
            # served read-only for diagnosis; no runtime cleanup or claims run.
            for executor in self.executors.values():
                executor.startup_cleanup()
            self.scheduler.start()
        except BaseException:
            self.coordinator.stop()
            raise

    def stop(self) -> None:
        self.scheduler.stop()
        self.coordinator.stop()
        self.store.close()


class BrokerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], broker: Broker) -> None:
        self.broker = broker
        self.upload_slots = threading.BoundedSemaphore(broker.config.max_concurrent_uploads)
        super().__init__(address, BrokerHandler)


class BrokerHandler(BaseHTTPRequestHandler):
    server: BrokerHTTPServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.broker.config.request_timeout_seconds)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"spark-broker {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        self._handle("GET")

    def do_HEAD(self) -> None:
        self._handle("HEAD")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        try:
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/health/live" and method in {"GET", "HEAD"}:
                self._json(HTTPStatus.OK, {"status": "live", "version": BROKER_VERSION}, head=method == "HEAD")
                return
            if not self._authorized():
                self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "valid bearer token required")
                return
            if path == "/health/ready" and method in {"GET", "HEAD"}:
                ready, scheduler_status = self._readiness()
                resources = scheduler_status.get("resources", {})
                self._json(
                    HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "ready" if ready else "not_ready", "resourceState": resources.get("resourceState")},
                    head=method == "HEAD",
                )
                return
            if path == "/v1/capabilities" and method == "GET":
                self._json(HTTPStatus.OK, self._capabilities())
                return
            if path == "/v1/status" and method == "GET":
                self._json(HTTPStatus.OK, {"protocolVersion": PROTOCOL_VERSION, "brokerId": self.server.broker.config.broker_id, **self.server.broker.scheduler.status()})
                return
            if path == "/v1/jobs" and method == "POST":
                self._submit_job()
                return
            if path == "/v1/artifacts" and method == "POST":
                self._upload_artifact(parsed.query)
                return
            parts = path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["v1", "jobs"] and _JOB_ID.fullmatch(parts[2]):
                self._job_route(method, parts)
                return
            if len(parts) >= 3 and parts[:2] == ["v1", "artifacts"] and _ARTIFACT_ID.fullmatch(parts[2]):
                self._artifact_route(method, parts)
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "route not found")
        except ContractError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": exc.as_dict()})
        except ArtifactError as exc:
            self._error(HTTPStatus.BAD_REQUEST, "artifact_error", str(exc))
        except IdempotencyConflict as exc:
            self._error(HTTPStatus.CONFLICT, "idempotency_conflict", str(exc))
        except QueueFull as exc:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "queue_full", str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except BaseException as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", f"request failed: {type(exc).__name__}")
        finally:
            # ThreadingHTTPServer may create a short-lived thread per request.
            # Store connections are thread-local, so release this handler's
            # connection before the thread is discarded.
            self.server.broker.store.close()

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.broker.config.token}"
        if len(supplied) != len(expected):
            return False
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))

    def _capabilities(self) -> dict[str, Any]:
        ready, scheduler_status = self._readiness()
        resource_state = scheduler_status.get("resources", {}).get("resourceState", "unavailable")
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "brokerVersion": BROKER_VERSION,
            "brokerId": self.server.broker.config.broker_id,
            "resourceState": resource_state,
            "capabilities": [
                {**capability, "available": ready}
                for capability in self.server.broker.scheduler.capabilities()
            ],
            "inferenceRoutes": {
                capability: executor.public_routes()
                for capability, executor in self.server.broker.executors.items()
                if hasattr(executor, "public_routes")
            },
            "limits": {
                "maxJsonBytes": self.server.broker.config.max_json_bytes,
                "maxArtifactBytes": self.server.broker.config.max_artifact_bytes,
                "maxHops": self.server.broker.config.max_hops,
                "maxPendingJobs": self.server.broker.config.max_pending_jobs,
                "maxStorageBytes": self.server.broker.config.max_storage_bytes,
                "maxConcurrentUploads": self.server.broker.config.max_concurrent_uploads,
            },
        }

    def _submit_job(self) -> None:
        body = self._json_body()
        request = validate_job_request(body, broker_id=self.server.broker.config.broker_id, max_hops=self.server.broker.config.max_hops)
        if request["capability"] not in self.server.broker.executors:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "capability_unavailable", "capability is not installed on this broker")
            return
        ready, _status = self._readiness()
        if not ready:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "broker_not_ready", "scheduler or resource coordinator is not ready")
            return
        for reference in request["inputs"]:
            artifact = self.server.broker.store.get_artifact(reference["artifactId"])
            if not artifact:
                self._error(HTTPStatus.UNPROCESSABLE_ENTITY, "artifact_not_found", f"input artifact {reference['artifactId']} does not exist")
                return
        job, created = self.server.broker.store.submit(request, max_pending_jobs=self.server.broker.config.max_pending_jobs)
        if created:
            self.server.broker.scheduler.notify()
        self._json(HTTPStatus.ACCEPTED if created else HTTPStatus.OK, self._public_job(job), headers={"Location": f"/v1/jobs/{job['id']}", "Idempotency-Replayed": "false" if created else "true"})

    def _readiness(self) -> tuple[bool, dict[str, Any]]:
        status = self.server.broker.scheduler.status()
        resources = status.get("resources", {})
        ready = bool(
            status["schedulerAlive"]
            and status["controlLoopAlive"]
            and resources.get("resourceState") == "ready"
        )
        return ready, status

    def _upload_artifact(self, query: str) -> None:
        if not self.server.upload_slots.acquire(blocking=False):
            raise QueueFull("concurrent artifact-upload limit reached")
        try:
            content_length = self._content_length(self.server.broker.config.max_artifact_bytes)
            params = urllib.parse.parse_qs(query, keep_blank_values=True)
            kind, role, media_type = validate_upload_metadata(
                params.get("kind", [""])[0], params.get("role", [""])[0], params.get("mediaType", [self.headers.get("Content-Type", "")])[0]
            )
            expected = self.headers.get("X-Content-SHA256")
            if not expected or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ContractError("invalid_hash", "X-Content-SHA256 is required and must be lowercase hex")
            artifact = self.server.broker.registry.import_stream(
                self.rfile,
                size=content_length,
                kind=kind,
                role=role,
                media_type=media_type,
                metadata={"uploadedBy": self.headers.get("X-Origin", "unknown")[:128]},
                expected_sha256=expected,
            )
            self._json(HTTPStatus.CREATED, {"protocolVersion": PROTOCOL_VERSION, "artifact": artifact}, headers={"Location": f"/v1/artifacts/{artifact['id']}"})
        finally:
            self.server.upload_slots.release()

    def _job_route(self, method: str, parts: list[str]) -> None:
        job_id = parts[2]
        if len(parts) == 3 and method == "GET":
            job = self.server.broker.store.get_job(job_id)
            if not job:
                self._error(HTTPStatus.NOT_FOUND, "job_not_found", "job not found")
                return
            self._json(HTTPStatus.OK, self._public_job(job))
            return
        if len(parts) == 4 and parts[3] == "events" and method == "GET":
            if not self.server.broker.store.get_job(job_id):
                self._error(HTTPStatus.NOT_FOUND, "job_not_found", "job not found")
                return
            self._json(HTTPStatus.OK, {"jobId": job_id, "events": self.server.broker.store.events(job_id)})
            return
        if len(parts) == 4 and parts[3] == "cancel" and method == "POST":
            job = self.server.broker.store.cancel(job_id)
            if not job:
                self._error(HTTPStatus.NOT_FOUND, "job_not_found", "job not found")
                return
            self.server.broker.scheduler.notify()
            self._json(HTTPStatus.ACCEPTED, self._public_job(job))
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "job route not found")

    def _artifact_route(self, method: str, parts: list[str]) -> None:
        artifact_id = parts[2]
        try:
            artifact, path = self.server.broker.registry.resolve(artifact_id, verify=True)
        except ArtifactNotFound:
            self._error(HTTPStatus.NOT_FOUND, "artifact_not_found", "artifact not found")
            return
        except ArtifactIntegrityError:
            self._error(HTTPStatus.CONFLICT, "artifact_integrity_failed", "registered artifact failed integrity verification")
            return
        if len(parts) == 3 and method == "GET":
            self._json(HTTPStatus.OK, {"protocolVersion": PROTOCOL_VERSION, "artifact": artifact})
            return
        if len(parts) == 4 and parts[3] == "content" and method in {"GET", "HEAD"}:
            start, end = 0, artifact["sizeBytes"] - 1
            range_header = self.headers.get("Range")
            partial = False
            if range_header:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if not match:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid_range", "only one explicit byte range is supported")
                    return
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else artifact["sizeBytes"] - 1
                if start > end or start >= artifact["sizeBytes"]:
                    self._error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "invalid_range", "requested range is outside the artifact")
                    return
                end = min(end, artifact["sizeBytes"] - 1)
                partial = True
            length = max(0, end - start + 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
            self.send_header("Content-Type", artifact["mediaType"])
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{artifact['sizeBytes']}")
            self.send_header("ETag", f'"sha256:{artifact["sha256"]}"')
            self.send_header("X-Content-SHA256", artifact["sha256"])
            self.send_header("Content-Disposition", f'attachment; filename="{artifact_id}"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if method == "GET":
                with path.open("rb") as reader:
                    reader.seek(start)
                    remaining = length
                    while remaining and (chunk := reader.read(min(1024 * 1024, remaining))):
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            return
        self._error(HTTPStatus.NOT_FOUND, "not_found", "artifact route not found")

    def _public_job(self, job: dict[str, Any]) -> dict[str, Any]:
        value = {key: val for key, val in job.items() if key != "request"}
        value["protocolVersion"] = PROTOCOL_VERSION
        value["links"] = {"self": f"/v1/jobs/{job['id']}", "events": f"/v1/jobs/{job['id']}/events", "cancel": f"/v1/jobs/{job['id']}/cancel"}
        return value

    def _json_body(self) -> dict[str, Any]:
        length = self._content_length(self.server.broker.config.max_json_bytes)
        if self.headers.get_content_type() != "application/json":
            raise ContractError("unsupported_media_type", "Content-Type must be application/json")
        data = self.rfile.read(length)
        try:
            value = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("invalid_json", "request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ContractError("invalid_json", "request body must be a JSON object")
        return value

    def _content_length(self, maximum: int) -> int:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise ContractError("length_required", "Content-Length is required")
        try:
            length = int(raw)
        except ValueError as exc:
            raise ContractError("invalid_length", "Content-Length is invalid") from exc
        if length < 0 or length > maximum:
            raise ContractError("request_too_large", f"Content-Length must not exceed {maximum}")
        return length

    def _json(self, status: HTTPStatus, value: Any, *, headers: dict[str, str] | None = None, head: bool = False) -> None:
        encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, content in (headers or {}).items():
            self.send_header(name, content)
        self.end_headers()
        if not head:
            self.wfile.write(encoded)

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._json(status, {"error": {"code": code, "message": message}})


def serve(config: Config) -> None:
    broker = Broker(config)
    server = BrokerHTTPServer((config.bind, config.port), broker)
    stop = threading.Event()

    def shutdown(_signum: int, _frame: Any) -> None:
        if not stop.is_set():
            stop.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    broker.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        broker.stop()


def main() -> None:
    serve(Config.from_env())


if __name__ == "__main__":
    main()
