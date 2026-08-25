from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .secure_files import SecureFileError, read_owner_secret


class ClientError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def read_credential(path: Path, label: str = "broker token") -> str:
    try:
        return read_owner_secret(path, label)
    except SecureFileError as exc:
        raise ValueError(str(exc)) from exc


class BrokerClient:
    def __init__(self, base_url: str, token: str, *, timeout: int = 60) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("broker URL must not contain credentials, a query, or a fragment")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("remote broker URLs must use HTTPS")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("broker URL must be HTTP loopback or HTTPS")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        handlers: list[Any] = [_NoRedirect()]
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
            handlers.insert(0, urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def capabilities(self) -> dict[str, Any]:
        return self._json("GET", "/v1/capabilities")

    def status(self) -> dict[str, Any]:
        return self._json("GET", "/v1/status")

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/jobs", request)

    def job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}")

    def events(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}/events")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}/cancel", b"", {"Content-Type": "application/json"})

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/artifacts/{urllib.parse.quote(artifact_id, safe='')}")

    def upload(self, path: Path, *, kind: str, role: str, media_type: str, origin: str) -> dict[str, Any]:
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError("client uploads are limited to 256 MiB; use a staged artifact importer for larger files")
        data = path.read_bytes()
        query = urllib.parse.urlencode({"kind": kind, "role": role, "mediaType": media_type})
        return self._request(
            "POST",
            f"/v1/artifacts?{query}",
            data,
            {"Content-Type": media_type, "X-Origin": origin, "X-Content-SHA256": hashlib.sha256(data).hexdigest()},
        )

    def upload_bytes(self, data: bytes, *, kind: str, role: str, media_type: str, origin: str) -> dict[str, Any]:
        query = urllib.parse.urlencode({"kind": kind, "role": role, "mediaType": media_type})
        return self._request(
            "POST",
            f"/v1/artifacts?{query}",
            data,
            {"Content-Type": media_type, "X-Origin": origin, "X-Content-SHA256": hashlib.sha256(data).hexdigest()},
        )

    def download(self, artifact_id: str, destination: Path) -> None:
        metadata = self.artifact(artifact_id)["artifact"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A fixed sidecar name lets concurrent downloads to the same target
        # corrupt or delete one another's partial file.  Keep the final rename
        # atomic while making each in-flight staging path private.
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        digest = hashlib.sha256()
        received = 0
        try:
            with temporary.open("wb") as target:
                while received < metadata["sizeBytes"]:
                    data, _headers = self.download_chunk(
                        artifact_id,
                        offset=received,
                        length=min(4 * 1024 * 1024, metadata["sizeBytes"] - received),
                        expected_sha256=metadata["sha256"],
                    )
                    if not data:
                        raise ClientError(0, "short_download", "artifact download ended before the registered size")
                    target.write(data)
                    digest.update(data)
                    received += len(data)
            if received != metadata["sizeBytes"] or digest.hexdigest() != metadata["sha256"]:
                raise ClientError(0, "hash_mismatch", "downloaded artifact failed integrity check")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(destination)

    def download_bytes(self, artifact_id: str, *, expected_sha256: str | None = None) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}/v1/artifacts/{urllib.parse.quote(artifact_id, safe='')}/content",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                data = response.read()
                expected = response.headers.get("X-Content-SHA256")
                if not expected or not re.fullmatch(r"[a-f0-9]{64}", expected):
                    raise ClientError(response.status, "missing_integrity", "artifact response omitted a valid content digest")
                if expected_sha256 is not None and expected != expected_sha256:
                    raise ClientError(response.status, "hash_mismatch", "artifact response digest did not match its metadata")
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ClientError(response.status, "hash_mismatch", "downloaded artifact failed integrity check")
                return data
        except urllib.error.HTTPError as exc:
            self._raise_http(exc)
        raise AssertionError("unreachable")

    def download_chunk(
        self,
        artifact_id: str,
        *,
        offset: int,
        length: int,
        expected_sha256: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        if offset < 0 or length < 1:
            raise ValueError("offset must be non-negative and length must be positive")
        request = urllib.request.Request(
            f"{self.base_url}/v1/artifacts/{urllib.parse.quote(artifact_id, safe='')}/content",
            headers={"Authorization": f"Bearer {self.token}", "Range": f"bytes={offset}-{offset + length - 1}"},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or not match or int(match.group(1)) != offset:
                    raise ClientError(response.status, "invalid_range", "artifact server returned a mismatched byte range")
                expected_length = int(match.group(2)) - int(match.group(1)) + 1
                if expected_length < 1 or expected_length > length:
                    raise ClientError(response.status, "invalid_range", "artifact server returned an invalid byte range length")
                data = response.read(expected_length + 1)
                if len(data) != expected_length:
                    raise ClientError(response.status, "invalid_range", "artifact server returned a byte range with the wrong length")
                artifact_digest = response.headers.get("X-Content-SHA256")
                if not artifact_digest or not re.fullmatch(r"[a-f0-9]{64}", artifact_digest):
                    raise ClientError(response.status, "missing_integrity", "artifact range omitted a valid content digest")
                if expected_sha256 is not None and artifact_digest != expected_sha256:
                    raise ClientError(response.status, "hash_mismatch", "artifact range digest did not match its metadata")
                return data, {key: value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            self._raise_http(exc)
        raise AssertionError("unreachable")

    def _json(self, method: str, path: str, value: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if value is None else json.dumps(value, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        return self._request(method, path, data, headers)

    def _request(self, method: str, path: str, data: bytes | None, headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method,
            headers={"Authorization": f"Bearer {self.token}", **headers},
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            self._raise_http(exc)
        except urllib.error.URLError as exc:
            raise ClientError(0, "connection_failed", str(exc.reason)) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_http(exc: urllib.error.HTTPError) -> None:
        try:
            body = json.load(exc)
            error = body.get("error", {})
            code = error.get("code", "http_error")
            message = error.get("message", str(exc))
        except (json.JSONDecodeError, AttributeError):
            code, message = "http_error", str(exc)
        exc.close()
        raise ClientError(exc.code, code, message) from exc
