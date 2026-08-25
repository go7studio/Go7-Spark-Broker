from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from . import PROTOCOL_VERSION
from .contract import ContractError
from .executors import ExecutionCancelled, ExecutionFailure, Executor
from .artifacts import ArtifactRegistry
from .store import Store
from .resources import AdmissionDeferred, ExecutionControl, ExecutionPlan, LeaseHandle, ResourceCoordinator


class Scheduler:
    def __init__(
        self,
        *,
        broker_id: str,
        store: Store,
        registry: ArtifactRegistry,
        executors: dict[str, Executor],
        coordinator: ResourceCoordinator | None = None,
    ) -> None:
        self.broker_id = broker_id
        self.store = store
        self.registry = registry
        self.executors = executors
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._control_thread: threading.Thread | None = None
        self._control_wake = threading.Event()
        self._state_lock = threading.Lock()
        self._active_job: str | None = None
        self._active_profile: str | None = None
        self._profile_loaded_at: float | None = None
        self._active_priority: int | None = None
        self._active_plan: ExecutionPlan | None = None
        self._active_control: ExecutionControl | None = None
        self.coordinator = coordinator

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="spark-scheduler", daemon=True)
        self._control_thread = threading.Thread(target=self._control_loop, name="spark-resource-control", daemon=True)
        self._thread.start()
        self._control_thread.start()

    def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        self._wake.set()
        self._control_wake.set()
        if self._thread:
            self._thread.join(timeout)
        if self._control_thread:
            self._control_thread.join(timeout)

    def notify(self) -> None:
        self._wake.set()
        self._control_wake.set()

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            value = {
                "activeJobId": self._active_job,
                "activeProfileId": self._active_profile,
                "profileLoadedAtMonotonic": self._profile_loaded_at,
                "schedulerAlive": bool(self._thread and self._thread.is_alive()),
                "controlLoopAlive": bool(self._control_thread and self._control_thread.is_alive()),
                "activePlan": self._active_plan.public() if self._active_plan else None,
            }
        if self.coordinator:
            value["resources"] = self.coordinator.status()
        return value

    def capabilities(self) -> list[dict[str, Any]]:
        return [self.executors[key].capability.public() for key in sorted(self.executors)]

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                job = self.store.claim_next(set(self.executors))
                if not job:
                    self._wake.wait(2.0)
                    self._wake.clear()
                    continue
                self._execute(job)
        finally:
            self.store.close()

    def _control_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._control_wake.wait(1.0)
                self._control_wake.clear()
                with self._state_lock:
                    active_priority = self._active_priority
                    plan = self._active_plan
                    control = self._active_control
                if active_priority is None or plan is None or control is None:
                    continue
                queued = self.store.peek_next(set(self.executors))
                if not queued or queued["priority"] <= active_priority:
                    continue
                if plan.preemption_mode == "checkpoint":
                    control.request_yield(f"higher_priority_job:{queued['id']}")
        finally:
            self.store.close()

    def _execute(self, job: dict[str, Any]) -> None:
        executor = self.executors[job["capability"]]
        lease: LeaseHandle | None = None
        plan: ExecutionPlan | None = None
        with self._state_lock:
            self._active_job = job["id"]
            self._active_priority = job["priority"]
        control = ExecutionControl(lambda: self._stop.is_set() or self.store.cancel_requested(job["id"]))
        with self._state_lock:
            self._active_control = control

        def stage(status: str, detail: dict[str, Any] | None = None) -> None:
            if control():
                raise ExecutionCancelled("job was cancelled")
            self.store.transition(job["id"], status, detail=detail, profile_id=plan.profile_id if plan else executor.capability.profile_id)

        try:
            self._check_deadline(job)
            # Pure validation and route selection happen before a model is
            # loaded, a container is stopped, or a throttle is requested.
            executor.validate_request(job, self.registry)
            routing_context = self.coordinator.routing_context() if self.coordinator else None
            plan = executor.plan(job, routing_context)
            with self._state_lock:
                self._active_plan = plan
            self.store.transition(job["id"], "loading", detail={"plan": plan.public()}, profile_id=plan.profile_id)
            if self.coordinator:
                lease = self.coordinator.acquire(job, plan, control)
            executor.activate_plan(plan, control)
            with self._state_lock:
                self._active_profile = plan.profile_id
                self._profile_loaded_at = time.monotonic()
            payload = executor.execute_plan(plan, job, self.registry, control, stage)
            if control():
                raise ExecutionCancelled("job was cancelled")
            if control.yield_requested():
                raise ExecutionFailure(
                    "yield_protocol_unsupported",
                    "executor yielded, but protocol 1.0 cannot represent a resumable training run safely",
                    retryable=False,
                )
            if self.coordinator:
                self.coordinator.release(lease)
                lease = None
            request = job["request"]
            result = {
                "protocolVersion": PROTOCOL_VERSION,
                "jobId": job["id"],
                "traceId": job["traceId"],
                "route": {"visitedSystems": [*request["visitedSystems"], self.broker_id], "hopCount": request["hopCount"] + 1},
                **payload,
            }
            result.setdefault("data", {})["executionPlan"] = plan.public()
            self.store.finish(job["id"], result)
        except AdmissionDeferred as exc:
            if exc.code == "cancelled":
                self.store.fail(job["id"], {"code": "cancelled", "message": str(exc), "retryable": False}, status="cancelled")
            else:
                self.store.defer(job["id"], detail={"code": exc.code, "message": str(exc), **exc.detail})
                self._wake.wait(exc.retry_after_seconds)
                self._wake.clear()
        except ExecutionCancelled as exc:
            self.store.fail(job["id"], {"code": "cancelled", "message": str(exc), "retryable": False}, status="cancelled")
        except ContractError as exc:
            self.store.fail(job["id"], {**exc.as_dict(), "retryable": False})
        except ExecutionFailure as exc:
            self.store.fail(job["id"], {"code": exc.code, "message": str(exc), "retryable": exc.retryable})
        except BaseException as exc:
            self.store.fail(job["id"], {"code": "internal_error", "message": f"executor failed: {type(exc).__name__}", "retryable": False})
        finally:
            if lease is not None and self.coordinator:
                try:
                    self.coordinator.release(lease)
                except AdmissionDeferred:
                    pass
            with self._state_lock:
                self._active_job = None
                self._active_priority = None
                self._active_plan = None
                self._active_control = None

    @staticmethod
    def _check_deadline(job: dict[str, Any]) -> None:
        deadline = job["request"].get("deadline")
        if deadline is None:
            return
        value = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
        if value.astimezone(timezone.utc) <= datetime.now(timezone.utc):
            raise ContractError("deadline_expired", "job deadline expired before admission", field="deadline")
