from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from .store import Store, utc_now


class ArtifactError(RuntimeError):
    pass


class ArtifactRegistry:
    def __init__(self, root: Path, store: Store, *, max_upload_bytes: int, max_storage_bytes: int | None = None) -> None:
        self.root = root.resolve()
        self.store = store
        self.max_upload_bytes = max_upload_bytes
        self.max_storage_bytes = max_storage_bytes
        self._quota_lock = threading.Lock()
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)

    def import_stream(
        self,
        stream: BinaryIO,
        *,
        size: int,
        kind: str,
        role: str,
        media_type: str,
        job_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        if size < 0 or size > self.max_upload_bytes:
            raise ArtifactError(f"artifact size must be between 0 and {self.max_upload_bytes} bytes")
        fd, staging_name = tempfile.mkstemp(prefix="upload-", dir=self.root / ".staging")
        digest = hashlib.sha256()
        read_bytes = 0
        try:
            with os.fdopen(fd, "wb") as target:
                while read_bytes < size:
                    chunk = stream.read(min(1024 * 1024, size - read_bytes))
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    read_bytes += len(chunk)
                target.flush()
                os.fsync(target.fileno())
            if read_bytes != size:
                raise ArtifactError(f"expected {size} bytes but received {read_bytes}")
            if expected_sha256 and digest.hexdigest() != expected_sha256:
                raise ArtifactError("uploaded artifact hash does not match X-Content-SHA256")
            return self._commit(Path(staging_name), digest.hexdigest(), size, kind, role, media_type, job_id, metadata, validation)
        except BaseException:
            Path(staging_name).unlink(missing_ok=True)
            raise

    def import_file(
        self,
        source: Path,
        *,
        kind: str,
        role: str,
        media_type: str,
        job_id: str | None,
        metadata: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = source.resolve(strict=True)
        size = source.stat().st_size
        if size > self.max_upload_bytes:
            raise ArtifactError(f"artifact exceeds {self.max_upload_bytes} bytes")
        fd, staging_name = tempfile.mkstemp(prefix="result-", dir=self.root / ".staging")
        digest = hashlib.sha256()
        try:
            with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            return self._commit(Path(staging_name), digest.hexdigest(), size, kind, role, media_type, job_id, metadata, validation)
        except BaseException:
            Path(staging_name).unlink(missing_ok=True)
            raise

    def _commit(
        self,
        staging: Path,
        sha256: str,
        size: int,
        kind: str,
        role: str,
        media_type: str,
        job_id: str | None,
        metadata: dict[str, Any] | None,
        validation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        artifact_id = f"art_{uuid.uuid4().hex}"
        relative = Path("objects") / sha256[:2] / artifact_id
        destination = self.root / relative
        artifact = {
            "id": artifact_id, "jobId": job_id, "kind": kind, "role": role,
            "mediaType": media_type, "sha256": sha256, "sizeBytes": size,
            "metadata": metadata or {}, "validation": validation or {}, "createdAt": utc_now(),
        }
        with self._quota_lock:
            if self.max_storage_bytes is not None and self.store.artifact_usage_bytes() + size > self.max_storage_bytes:
                staging.unlink(missing_ok=True)
                raise ArtifactError("artifact storage quota would be exceeded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            try:
                self.store.add_artifact(artifact, str(relative))
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
        return artifact

    def resolve(self, artifact_id: str, *, verify: bool = False) -> tuple[dict[str, Any], Path]:
        artifact = self.store.get_artifact(artifact_id)
        if not artifact:
            raise ArtifactError("artifact not found")
        relative = Path(artifact.pop("_relativePath"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactError("artifact registry contains an unsafe path")
        path = (self.root / relative).resolve(strict=True)
        if self.root not in path.parents:
            raise ArtifactError("artifact path escaped registry root")
        if verify:
            digest = hashlib.sha256()
            with path.open("rb") as reader:
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != artifact["sha256"] or path.stat().st_size != artifact["sizeBytes"]:
                raise ArtifactError("artifact integrity check failed")
        return artifact, path

    def clean_staging(self) -> None:
        for path in (self.root / ".staging").iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
