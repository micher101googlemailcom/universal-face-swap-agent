"""Offline regression tests using synthetic files and standard-library mocks."""

import contextlib
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import inventory


class ScandirProxy:
    """Preserve both iterator and context-manager behavior of os.scandir."""

    def __init__(self, entries, iterator):
        self.entries = entries
        self.iterator = iter(iterator)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.entries.close()


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

    def run_inventory(self, *extra_args):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        args = ["--scan", str(self.root), "--output", str(self.output), *extra_args]
        with patch.object(inventory, "optional", return_value=(None, "ImportError")), \
                contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            inventory.main(args)

    def write_provenance(self, entries):
        path = self.base / "private-provenance.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return str(path)

    @contextlib.contextmanager
    def patch_entries(self, transform):
        real_scandir = os.scandir

        def scandir(path):
            entries = real_scandir(path)
            return ScandirProxy(entries, (transform(entry) for entry in entries))

        with patch.object(inventory.os, "scandir", side_effect=scandir):
            yield

    def assert_failed_without_reports(self, error_type, private_name):
        with self.assertRaises(SystemExit) as caught:
            self.run_inventory()
        self.assertEqual(caught.exception.code, 2)
        message = self.stderr.getvalue()
        self.assertIn("root-1", message)
        self.assertIn(error_type.__name__, message)
        self.assertIn("inventory not written", message)
        for private_text in (str(self.base), private_name, "private error detail"):
            self.assertNotIn(private_text, message)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def make_symlink(self, link, target, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("symlink creation unavailable: " + type(exc).__name__)

    def assert_output_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_inventory()
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertNotIn(str(self.base), self.stderr.getvalue())
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_entry_classification_errors_abort_without_partial_reports(self):
        nested = self.root / "private-nested-directory"
        nested.mkdir()
        for error_type in (PermissionError, OSError, FileNotFoundError):
            with self.subTest(error=error_type.__name__):
                self.output = self.base / ("reports-" + error_type.__name__)
                error = error_type(errno.EIO, "private error detail", str(nested))

                def transform(entry):
                    if entry.name != nested.name:
                        return entry
                    proxy = Mock(wraps=entry, name=entry.name)
                    proxy.name, proxy.path = entry.name, entry.path
                    proxy.is_dir.side_effect = error
                    proxy.stat.side_effect = error
                    return proxy

                with self.patch_entries(transform):
                    self.assert_failed_without_reports(error_type, nested.name)

    def test_scandir_iteration_error_aborts_without_partial_reports(self):
        error = OSError(errno.EIO, "private error detail", str(self.root))
        real_scandir = os.scandir

        def scandir(path):
            entries = real_scandir(path)

            def interrupted():
                yield next(entries)
                raise error

            return ScandirProxy(entries, interrupted())

        with patch.object(inventory.os, "scandir", side_effect=scandir):
            self.assert_failed_without_reports(OSError, self.root.name)

    def test_windows_reparse_entries_are_never_traversed_or_read(self):
        junction = self.root / "private-junction"
        junction.mkdir()
        (junction / "outside.engine").write_bytes(b"outside selected tree")
        linked_file = self.root / "private-reparse.engine"
        linked_file.write_bytes(b"linked model")
        real_scandir = os.scandir
        real_digest = inventory.digest

        def transform(entry):
            if Path(entry.path) not in (junction, linked_file):
                return entry
            proxy = Mock(wraps=entry, name=entry.name)
            proxy.name, proxy.path = entry.name, entry.path
            proxy.stat.return_value = SimpleNamespace(
                st_mode=entry.stat(follow_symlinks=False).st_mode,
                st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return proxy

        def guarded_scandir(path):
            self.assertNotEqual(Path(path), junction, "must not enumerate a junction")
            return real_scandir(path)

        def guarded_digest(path):
            self.assertEqual(path, self.source, "must not read reparse-point assets")
            return real_digest(path)

        with patch.object(inventory.os, "scandir", side_effect=guarded_scandir), \
                patch.object(inventory, "digest", side_effect=guarded_digest), self.patch_entries(transform):
            self.run_inventory()
        data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
        self.assertEqual([a["sha256"] for a in data["assets"]], [self.sha256])
        self.assertEqual(linked_file.read_bytes(), b"linked model")

    @unittest.skipUnless(os.name == "nt", "requires Windows NTFS junctions")
    def test_native_windows_junction_escape_and_cycle_are_pruned(self):
        outside = self.base / "private-outside"
        outside.mkdir()
        external = outside / "external.engine"
        external.write_bytes(b"outside selected tree")
        junctions = [self.root / "private-junction", self.root / "private-cycle"]
        for link, target in zip(junctions, (outside, self.root)):
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                           check=True, capture_output=True)
            self.addCleanup(link.rmdir)
            self.assertTrue(link.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        real_scandir = os.scandir

        def scandir(path):
            self.assertNotIn(Path(path), junctions, "must not enumerate a junction or cycle")
            return real_scandir(path)

        with patch.object(inventory.os, "scandir", side_effect=scandir):
            self.run_inventory()
        data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
        self.assertEqual([a["sha256"] for a in data["assets"]], [self.sha256])
        self.assertEqual(external.read_bytes(), b"outside selected tree")

    def test_symlinked_files_directories_and_cycles_are_still_skipped(self):
        outside = self.base / "private-outside"
        outside.mkdir()
        external = outside / "external.engine"
        external.write_bytes(b"outside selected tree")
        self.make_symlink(self.root / "linked.engine", external)
        self.make_symlink(self.root / "linked-directory", outside, directory=True)
        self.make_symlink(self.root / "cycle", self.root, directory=True)
        self.make_symlink(self.root / "dangling.engine", outside / "absent.engine")
        self.run_inventory()
        data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
        self.assertEqual([a["sha256"] for a in data["assets"]], [self.sha256])
        self.assertEqual(external.read_bytes(), b"outside selected tree")

    def test_dangling_output_symlinks_are_rejected_before_writing(self):
        for name, other in (("inventory.json", "inventory.md"), ("inventory.md", "inventory.json")):
            with self.subTest(name=name):
                self.output = self.base / ("reports-" + name)
                self.output.mkdir()
                target = self.root / ("private-redirect-" + name)
                link = self.output / name
                self.make_symlink(link, target)
                self.assert_output_rejected()
                self.assertTrue(link.is_symlink())
                self.assertFalse(target.exists())
                self.assertFalse((self.output / other).exists())

    def test_existing_output_files_and_links_are_not_overwritten(self):
        for name in ("inventory.json", "inventory.md"):
            for kind in ("file", "symlink", "directory"):
                with self.subTest(name=name, kind=kind):
                    self.output = self.base / ("reports-" + name + "-" + kind)
                    self.output.mkdir()
                    target = self.base / ("private-target-" + name + "-" + kind)
                    target.write_bytes(b"keep existing data")
                    path = self.output / name
                    if kind == "symlink":
                        self.make_symlink(path, target)
                    elif kind == "directory":
                        path.mkdir()
                    else:
                        path.write_bytes(b"keep existing data")
                    self.assert_output_rejected()
                    self.assertEqual(target.read_bytes(), b"keep existing data")
                    self.assertEqual(list(self.output.iterdir()), [path])
                    if kind != "directory":
                        self.assertEqual(path.read_bytes(), b"keep existing data")

    def test_output_created_after_preflight_is_not_followed_or_overwritten(self):
        real_scan = inventory.scan
        for name, other in (("inventory.json", "inventory.md"), ("inventory.md", "inventory.json")):
            for kind in ("dangling-symlink", "file"):
                with self.subTest(name=name, kind=kind):
                    self.output = self.base / ("reports-late-" + name + "-" + kind)
                    target = self.root / ("private-late-" + name + "-" + kind)
                    path = self.output / name

                    def scan(*args):
                        assets = real_scan(*args)
                        self.output.mkdir()
                        if kind == "dangling-symlink":
                            self.make_symlink(path, target)
                        else:
                            path.write_bytes(b"concurrent writer")
                        return assets

                    with patch.object(inventory, "scan", side_effect=scan):
                        self.assert_output_rejected()
                    self.assertFalse(target.exists())
                    if kind == "dangling-symlink":
                        self.assertTrue(path.is_symlink())
                    else:
                        self.assertEqual(path.read_bytes(), b"concurrent writer")
                    # Reserving one output before the other fails must not publish inventory data.
                    if (self.output / other).exists():
                        self.assertEqual((self.output / other).read_bytes(), b"")

    def test_regular_nested_assets_and_explicit_roots_keep_stable_order(self):
        for name in ("z-directory", "a-directory"):
            directory = self.root / name
            directory.mkdir()
            (directory / "nested.engine").write_bytes(name.encode())
        second = self.base / "private-second-root"
        second.mkdir()
        (second / "second.engine").write_bytes(b"second")
        unselected = self.base / "private-unselected"
        unselected.mkdir()
        (unselected / "excluded.engine").write_bytes(b"not selected")
        self.run_inventory("--scan", str(second))
        data = json.loads((self.output / "inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(data["scan_roots"], ["root-1", "root-2"])
        self.assertEqual([a["scan_root"] for a in data["assets"]], ["root-1"] * 3 + ["root-2"])
        self.assertEqual([a["sha256"] for a in data["assets"]], [
            hashlib.sha256(payload).hexdigest()
            for payload in (self.payload, b"a-directory", b"z-directory", b"second")])

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
