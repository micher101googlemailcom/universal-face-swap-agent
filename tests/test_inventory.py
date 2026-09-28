"""Offline regression tests using synthetic files and standard-library mocks."""

import contextlib
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "private-models"
        self.root.mkdir()
        self.output = self.base / "reports"
        self.source = self.root / "private-model_sm120.engine"
        self.payload = b"synthetic inventory regression fixture"
        self.source.write_bytes(self.payload)
        self.sha256 = hashlib.sha256(self.payload).hexdigest()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def run_inventory(self, *extra_args):
        args = ["--scan", str(self.root), "--output", str(self.output), *extra_args]
        with patch.object(inventory, "optional", return_value=(None, "ImportError")), \
                contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            inventory.main(args)

    def write_provenance(self, entries):
        path = self.base / "private-provenance.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return str(path)

    def assert_traversal_failure(self, directory, error, label="root-1", extra_args=()):
        real_scandir = os.scandir

        def scandir(path):
            if Path(path) == directory:
                raise error
            return real_scandir(path)

        with patch.object(inventory.os, "scandir", side_effect=scandir):
            with self.assertRaises(SystemExit) as caught:
                self.run_inventory(*extra_args)

        self.assertEqual(caught.exception.code, 2)
        message = self.stderr.getvalue()
        self.assertIn(label, message)
        self.assertIn(type(error).__name__, message)
        self.assertIn("inventory not written", message)
        for private_text in (str(self.base), directory.name, "private error detail"):
            self.assertNotIn(private_text, message)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_unreadable_scan_root_aborts_without_reports(self):
        error = PermissionError(errno.EACCES, "private error detail", str(self.root))
        self.assert_traversal_failure(self.root, error)

    def test_nested_directory_io_error_aborts_without_partial_reports(self):
        nested = self.root / "private-nested-directory"
        nested.mkdir()
        hidden = nested / "private-hidden.engine"
        hidden.write_bytes(b"omitted model")
        error = OSError(errno.EIO, "private error detail", str(nested))
        self.assert_traversal_failure(nested, error)
        self.assertEqual(hidden.read_bytes(), b"omitted model")

    def test_failure_in_later_scan_root_aborts_without_partial_reports(self):
        second = self.base / "private-second-root"
        second.mkdir()
        error = PermissionError(errno.EACCES, "private error detail", str(second))
        self.assert_traversal_failure(second, error, "root-2", ("--scan", str(second)))

    def test_provenance_hashes_match_regardless_of_case(self):
        provenance = {"source": "owner supplied", "license": "review pending", "note": "fixture"}
        mixed = "".join(c.upper() if i % 2 else c for i, c in enumerate(self.sha256))
        for case, key in (("lower", self.sha256), ("upper", self.sha256.upper()), ("mixed", mixed)):
            with self.subTest(case=case):
                self.output = self.base / ("reports-" + case)
                path = self.write_provenance({key: provenance})
                self.run_inventory("--provenance", path)
                data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
                self.assertEqual(len(data["assets"]), 1)
                asset = data["assets"][0]
                self.assertEqual(asset["sha256"], self.sha256)
                self.assertEqual(asset["provenance"], provenance)
                self.assertEqual(asset["size_bytes"], len(self.payload))
                self.assertTrue(asset["engine_hints"]["sm120_in_filename"])
                for filename in ("inventory.json", "inventory.md"):
                    report = (self.output / filename).read_text(encoding="utf-8")
                    self.assertNotIn(str(self.root), report)
                    self.assertNotIn(self.source.name, report)
                self.assertEqual(self.source.read_bytes(), self.payload)

    def test_provenance_case_collisions_are_rejected(self):
        path = self.write_provenance({self.sha256: {"source": "first"},
                                      self.sha256.upper(): {"source": "second"}})
        with self.assertRaises(SystemExit) as caught:
            self.run_inventory("--provenance", path)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("duplicate SHA-256", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_invalid_provenance_hash_is_still_rejected(self):
        path = self.write_provenance({self.sha256[:-1]: {"source": "fixture"}})
        with self.assertRaises(SystemExit) as caught:
            self.run_inventory("--provenance", path)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("must map SHA-256 hashes to objects", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_uppercase_hash_does_not_bypass_provenance_sanitization(self):
        path = self.write_provenance({self.sha256.upper(): {"source": "C:\\private\\source"}})
        with self.assertRaises(SystemExit) as caught:
            self.run_inventory("--provenance", path)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("only sanitized", self.stderr.getvalue())
        self.assertNotIn("C:\\private\\source", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_file_read_failure_is_still_recorded_without_private_paths(self):
        error = PermissionError(errno.EACCES, "private error detail", str(self.source))
        with patch.object(inventory, "digest", side_effect=error):
            self.run_inventory()
        data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["assets"]), 1)
        self.assertEqual(data["assets"][0]["error"], "PermissionError")
        for filename in ("inventory.json", "inventory.md"):
            report = (self.output / filename).read_text(encoding="utf-8")
            self.assertIn("PermissionError", report)
            self.assertNotIn(self.source.name, report)
            self.assertNotIn(str(self.root), report)
            self.assertNotIn("private error detail", report)
        self.assertEqual(self.source.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
