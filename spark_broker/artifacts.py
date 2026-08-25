from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from .store import Store, utc_now


class ArtifactError(RuntimeError):
    pass


class ArtifactNotFound(ArtifactError):
    pass


class ArtifactIntegrityError(ArtifactError):
    pass


class ArtifactRegistry:
    def __init__(self, root: Path, store: Store, *, max_upload_bytes: int, max_storage_bytes: int | None = None) -> None:
        self.root = root.resolve()
        self.store = store
        self.max_upload_bytes = max_upload_bytes
        self.max_storage_bytes = max_storage_bytes
        self._quota_lock = threading.Lock()
        self._verification_lock = threading.Lock()
        self._verified: dict[str, tuple[int, int, int, int, int, str]] = {}
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)
        (self.root / ".orphaned").mkdir(parents=True, exist_ok=True)

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
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            source_descriptor = os.open(source, flags)
        except OSError as exc:
            raise ArtifactError("artifact source cannot be opened safely") from exc
        try:
            source_stat = os.fstat(source_descriptor)
        except OSError as exc:
            os.close(source_descriptor)
            raise ArtifactError("artifact source cannot be inspected safely") from exc
        if not stat.S_ISREG(source_stat.st_mode):
            os.close(source_descriptor)
            raise ArtifactError("artifact source must be a regular file")
        size = source_stat.st_size
        if size > self.max_upload_bytes:
            os.close(source_descriptor)
            raise ArtifactError(f"artifact exceeds {self.max_upload_bytes} bytes")
        fd, staging_name = tempfile.mkstemp(prefix="result-", dir=self.root / ".staging")
        digest = hashlib.sha256()
        copied_size = 0
        try:
            with os.fdopen(source_descriptor, "rb") as reader, os.fdopen(fd, "wb") as writer:
                source_descriptor = -1
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                    digest.update(chunk)
                    copied_size += len(chunk)
                final_source_stat = os.fstat(reader.fileno())
                writer.flush()
                os.fsync(writer.fileno())
            if (
                copied_size != size
                or final_source_stat.st_size != source_stat.st_size
                or final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
                or final_source_stat.st_ctime_ns != source_stat.st_ctime_ns
                or final_source_stat.st_ino != source_stat.st_ino
                or final_source_stat.st_dev != source_stat.st_dev
            ):
                raise ArtifactError("artifact source changed during import")
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ArtifactError("artifact source changed after validation")
            return self._commit(
                Path(staging_name), digest.hexdigest(), copied_size,
                kind, role, media_type, job_id, metadata, validation,
            )
        except BaseException:
            Path(staging_name).unlink(missing_ok=True)
            raise
        finally:
            if source_descriptor >= 0:
                os.close(source_descriptor)

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
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            try:
                self.store.add_artifact(artifact, str(relative))
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
            destination_stat = destination.stat()
            with self._verification_lock:
                self._verified[artifact_id] = self._verification_key(destination_stat, sha256)
        return artifact

    def resolve(self, artifact_id: str, *, verify: bool = False) -> tuple[dict[str, Any], Path]:
        artifact = self.store.get_artifact(artifact_id)
        if not artifact:
            raise ArtifactNotFound("artifact not found")
        relative = Path(artifact.pop("_relativePath"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactIntegrityError("artifact registry contains an unsafe path")
        path = self.root / relative
        if verify:
            self._verify(artifact_id, artifact, path)
        return artifact, path

    def clean_staging(self) -> None:
        for path in (self.root / ".staging").iterdir():
            if path.is_file() or path.is_symlink():
                self._quarantine(path, Path("staging") / path.name)
        known = {Path(value) for value in self.store.artifact_relative_paths()}
        objects = self.root / "objects"
        for path in list(objects.rglob("*")):
            if path.is_file() and path.relative_to(self.root) not in known:
                self._quarantine(path, Path("objects") / path.relative_to(objects))

    @staticmethod
    def _verification_key(metadata: os.stat_result, digest: str) -> tuple[int, int, int, int, int, str]:
        return (
            metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns, digest,
        )

    def _verify(self, artifact_id: str, artifact: dict[str, Any], path: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ArtifactIntegrityError("artifact object cannot be opened safely") from exc
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIntegrityError("artifact object is not a regular file")
            expected = artifact["sha256"]
            key = self._verification_key(before, expected)
            with self._verification_lock:
                if self._verified.get(artifact_id) == key:
                    return
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                self._verification_key(before, expected) != self._verification_key(after, expected)
                or after.st_size != artifact["sizeBytes"]
                or digest.hexdigest() != expected
            ):
                raise ArtifactIntegrityError("artifact integrity check failed")
            with self._verification_lock:
                self._verified[artifact_id] = self._verification_key(after, expected)
        finally:
            os.close(descriptor)

    def _quarantine(self, source: Path, relative: Path) -> None:
        destination = self.root / ".orphaned" / relative
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists() or destination.is_symlink():
            destination = destination.with_name(f"{destination.name}-{uuid.uuid4().hex}")
        os.replace(source, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
