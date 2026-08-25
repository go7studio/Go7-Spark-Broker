from __future__ import annotations

import os
import stat
from pathlib import Path


class SecureFileError(ValueError):
    pass


def read_owner_file(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    """Read one service-owned regular file without following a final symlink."""

    if not path.is_absolute():
        raise SecureFileError(f"{label} path must be absolute")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecureFileError(f"{label} file cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SecureFileError(f"{label} file must be regular")
        if metadata.st_uid != os.geteuid():
            raise SecureFileError(f"{label} file must be owned by the current account")
        if metadata.st_mode & 0o077:
            raise SecureFileError(f"{label} file must be mode 0600 or stricter")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            value = stream.read(maximum_bytes + 1)
        if len(value) > maximum_bytes:
            raise SecureFileError(f"{label} file is too large")
        return value
    finally:
        os.close(descriptor)


def read_owner_text(path: Path, label: str, *, maximum_bytes: int) -> str:
    try:
        return read_owner_file(path, label, maximum_bytes=maximum_bytes).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SecureFileError(f"{label} file must be UTF-8") from exc


def read_owner_secret(
    path: Path,
    label: str,
    *,
    minimum_length: int = 32,
    maximum_bytes: int = 4096,
) -> str:
    value = read_owner_text(path, label, maximum_bytes=maximum_bytes).strip()
    if len(value) < minimum_length or "\n" in value or "\r" in value:
        raise SecureFileError(f"{label} is missing, too short, or not a single value")
    return value
