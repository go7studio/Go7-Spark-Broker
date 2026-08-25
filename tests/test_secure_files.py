from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from spark_broker.secure_files import SecureFileError, read_owner_secret, read_owner_text


class SecureFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def file(self, name: str, value: bytes, mode: int = 0o600) -> Path:
        path = self.root / name
        path.write_bytes(value)
        path.chmod(mode)
        return path

    def test_owner_only_regular_file_is_read(self) -> None:
        value = self.file("token", b"t" * 32)
        self.assertEqual(read_owner_secret(value, "token"), "t" * 32)

    def test_symlink_fifo_and_permissive_file_are_rejected(self) -> None:
        target = self.file("target", b"t" * 32)
        link = self.root / "link"
        link.symlink_to(target)
        with self.assertRaisesRegex(SecureFileError, "opened safely"):
            read_owner_secret(link, "token")

        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(SecureFileError, "regular"):
            read_owner_text(fifo, "config", maximum_bytes=1024)

        target.chmod(0o640)
        with self.assertRaisesRegex(SecureFileError, "0600"):
            read_owner_secret(target, "token")

    def test_size_utf8_and_single_value_limits_fail_closed(self) -> None:
        oversized = self.file("oversized", b"x" * 17)
        with self.assertRaisesRegex(SecureFileError, "too large"):
            read_owner_text(oversized, "config", maximum_bytes=16)
        invalid = self.file("invalid", b"\xff")
        with self.assertRaisesRegex(SecureFileError, "UTF-8"):
            read_owner_text(invalid, "config", maximum_bytes=16)
        multiline = self.file("multiline", b"a" * 32 + b"\n" + b"b")
        with self.assertRaisesRegex(SecureFileError, "single value"):
            read_owner_secret(multiline, "token")


if __name__ == "__main__":
    unittest.main()
