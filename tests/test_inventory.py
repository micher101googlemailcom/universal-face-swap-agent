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
            if entry.name not in (junction.name, linked_file.name):
                return entry
            proxy = Mock(wraps=entry, name=entry.name)
            proxy.name, proxy.path = entry.name, entry.path
            proxy.stat.return_value = SimpleNamespace(
                st_mode=entry.stat(follow_symlinks=False).st_mode,
                st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
            return proxy

        def guarded_scandir(path):
            self.assertFalse(os.path.samestat(os.fstat(path), junction.stat()) if isinstance(path, int)
                             else Path(path) == junction, "must not enumerate a junction")
            return real_scandir(path)

        def guarded_digest(path):
            self.assertTrue(os.path.samestat(os.fstat(path.fileno()), self.source.stat()),
                            "must not read reparse-point assets")
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
                            self.assertTrue(path.is_symlink())
                            self.assertFalse(target.exists())
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
            if (os.path.samestat(os.fstat(path), directory.stat()) if isinstance(path, int)
                    else Path(path) == directory):
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


    def read_assets(self, output=None):
        return json.loads(((output or self.output) / "inventory.json").read_text(encoding="utf-8"))["assets"]

    def test_exact_duplicate_provenance_keys_are_rejected(self):
        path = self.base / "duplicate.json"
        path.write_text('{"' + self.sha256 + '": {"source":"first"}, "' +
                        self.sha256 + '": {"source":"second"}}', encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_inventory("--provenance", str(path))
        self.assertIn("duplicate", self.stderr.getvalue())
        self.assertFalse(self.output.exists())

    def swap_with_link(self, path, outside):
        """A held Windows directory/file must deny rename; POSIX pins the old inode."""
        saved = path.with_name(path.name + "-saved")
        try:
            path.rename(saved)
        except PermissionError:
            if os.name != "nt":
                raise
            return saved, False
        self.make_symlink(path, outside, directory=outside.is_dir())
        return saved, True

    def test_queued_directory_swap_is_rejected(self):
        nested = self.root / "nested"
        nested.mkdir()
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "external.engine").write_bytes(b"never read")
        real_file = inventory.secure_io.Directory.file
        swapped = False

        @contextlib.contextmanager
        def file(directory, name, create=False):
            nonlocal swapped
            if name == self.source.name and not swapped:
                nested.rename(self.root / "original-nested")
                self.make_symlink(nested, outside, directory=True)
                swapped = True
            with real_file(directory, name, create) as stream:
                yield stream

        with patch.object(inventory.secure_io.Directory, "file", file):
            with self.assertRaises(SystemExit):
                self.run_inventory()
        self.assertTrue(swapped)
        self.assertFalse(self.output.exists())
        self.assertEqual((outside / "external.engine").read_bytes(), b"never read")

    def test_open_directory_swap_cannot_redirect_enumeration(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "external.engine").write_bytes(b"never read")
        real_entries = inventory.secure_io.Directory.entries
        attempted = False

        def entries(directory):
            nonlocal attempted
            if directory.path == self.root and not attempted:
                attempted = True
                self.swap_with_link(self.root, outside)
            return real_entries(directory)

        with patch.object(inventory.secure_io.Directory, "entries", entries):
            self.run_inventory()
        self.assertTrue(attempted)
        self.assertEqual([a["sha256"] for a in self.read_assets()], [self.sha256])

    def test_asset_swap_before_open_never_reads_external_data(self):
        outside = self.base / "external.engine"
        outside.write_bytes(b"never read")
        real_file = inventory.secure_io.Directory.file
        attempted = False

        @contextlib.contextmanager
        def file(directory, name, create=False):
            nonlocal attempted
            if name == self.source.name and not attempted:
                attempted = True
                self.source.rename(self.root / "original.engine")
                self.make_symlink(self.source, outside)
            with real_file(directory, name, create) as stream:
                yield stream

        with patch.object(inventory.secure_io.Directory, "file", file), \
                patch.object(inventory, "digest", side_effect=AssertionError("must not hash a link")):
            self.run_inventory()
        asset, = self.read_assets()
        self.assertIn("error", asset)
        self.assertNotIn("sha256", asset)

    def test_hash_and_metadata_use_same_open_file_after_swap(self):
        source = self.root / "model.onnx"
        source.write_bytes(b"original model")
        outside = self.base / "external.onnx"
        outside.write_bytes(b"never read")
        real_digest = inventory.digest
        metadata = Mock(side_effect=lambda stream, module: {"payload": stream.read().decode()})

        def digest(stream):
            if os.path.samestat(os.fstat(stream.fileno()), source.stat()):
                self.swap_with_link(source, outside)
            result = real_digest(stream)
            stream.seek(0)
            return result

        with patch.object(inventory, "digest", side_effect=digest), \
                patch.object(inventory, "onnx_details", metadata):
            assets = inventory.scan(self.root, "root-1", {"onnx": object(), "safetensors": None}, {})
        asset = next(a for a in assets if a["type"] == "onnx")
        self.assertEqual(asset["sha256"], hashlib.sha256(b"original model").hexdigest())
        self.assertEqual(asset["onnx"], {"payload": "original model"})
        self.assertEqual(metadata.call_count, 1)

    def test_output_ancestor_swap_cannot_redirect_writes(self):
        parent = self.base / "output-parent"
        parent.mkdir()
        self.output = parent / "reports"
        real_scan = inventory.scan
        result_path = self.output

        def scan(*args):
            nonlocal result_path
            assets = real_scan(*args)
            saved, swapped = self.swap_with_link(parent, self.root)
            if swapped:
                result_path = saved / "reports"
            return assets

        with patch.object(inventory, "scan", side_effect=scan):
            self.run_inventory()
        self.assertFalse((self.root / "reports").exists())
        self.assertEqual([a["sha256"] for a in self.read_assets(result_path)], [self.sha256])

    def test_late_missing_output_component_link_is_rejected(self):
        real_scan = inventory.scan

        def scan(*args):
            assets = real_scan(*args)
            self.make_symlink(self.output, self.root, directory=True)
            return assets

        with patch.object(inventory, "scan", side_effect=scan):
            self.assert_output_rejected()
        self.assertFalse((self.root / "inventory.json").exists())
        self.assertFalse((self.root / "inventory.md").exists())

    def test_git_metadata_links_are_not_followed(self):
        outside = self.base / "external-git"
        outside.mkdir()
        (outside / "HEAD").write_text("a" * 40, encoding="ascii")
        self.make_symlink(self.root / ".git", outside, directory=True)
        self.assertNotIn("revision", inventory.version_hint(self.root))

    @unittest.skipUnless(os.name == "nt", "requires Windows NTFS junctions")
    def test_git_junction_is_not_followed(self):
        outside = self.base / "external-git"
        outside.mkdir()
        (outside / "HEAD").write_text("a" * 40, encoding="ascii")
        link = self.root / ".git"
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True)
        self.addCleanup(link.rmdir)
        self.assertNotIn("revision", inventory.version_hint(self.root))

    def test_git_head_and_ref_links_are_not_followed(self):
        git = self.root / ".git"
        git.mkdir()
        outside = self.base / "external-head"
        outside.write_text("a" * 40, encoding="ascii")
        self.make_symlink(git / "HEAD", outside)
        self.assertNotIn("revision", inventory.version_hint(self.root))
        (git / "HEAD").unlink()
        (git / "HEAD").write_text("ref: refs/heads/main", encoding="ascii")
        (git / "refs" / "heads").mkdir(parents=True)
        self.make_symlink(git / "refs" / "heads" / "main", outside)
        self.assertNotIn("revision", inventory.version_hint(self.root))

    def test_regular_git_metadata_is_preserved(self):
        git = self.root / ".git"
        git.mkdir()
        head = git / "HEAD"
        head.write_text("a" * 40, encoding="ascii")
        self.assertEqual(inventory.version_hint(self.root)["revision"], "a" * 40)
        head.write_text("ref: refs/heads/main", encoding="ascii")
        (git / "refs" / "heads").mkdir(parents=True)
        ref = git / "refs" / "heads" / "main"
        ref.write_text("b" * 40, encoding="ascii")
        self.assertEqual(inventory.version_hint(self.root)["revision"], "b" * 40)
        ref.unlink()
        (git / "packed-refs").write_text("c" * 40 + " refs/heads/main\n", encoding="ascii")
        self.assertEqual(inventory.version_hint(self.root)["revision"], "c" * 40)

    def test_safetensors_metadata_reads_verified_stream_without_reopening(self):
        import struct
        source = self.root / "model.safetensors"
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        source.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        module = Mock()
        with source.open("rb") as stream:
            data = inventory.safetensors_details(stream, module)
        module.safe_open.assert_not_called()
        self.assertEqual(data, {"tensors": [{"key": "weight", "dtype": "F32", "shape": [1]}]})

    @unittest.skipUnless(os.name == "nt", "requires Windows NTFS junctions")
    def test_late_output_junction_is_rejected(self):
        real_scan = inventory.scan
        def scan(*args):
            assets = real_scan(*args)
            subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(self.output), str(self.root)],
                           check=True, capture_output=True)
            self.addCleanup(self.output.rmdir)
            return assets
        with patch.object(inventory, "scan", side_effect=scan):
            self.assert_output_rejected()
        self.assertFalse((self.root / "inventory.json").exists())
        self.assertFalse((self.root / "inventory.md").exists())

    def test_safetensors_invalid_headers_do_not_produce_metadata(self):
        import struct
        source = self.root / "invalid.safetensors"
        for header in (
                {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
                {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}},
                {"weight": {"dtype": "unknown", "shape": [1], "data_offsets": [0, 4]}}):
            with self.subTest(header=header):
                raw = json.dumps(header).encode()
                source.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
                with source.open("rb") as stream, self.assertRaises(ValueError):
                    inventory.safetensors_details(stream, object())

    def test_external_onnx_tensors_in_all_message_locations(self):
        try:
            import onnx
        except ImportError:
            if os.environ.get("INVENTORY_REQUIRE_ONNX") == "1":
                self.fail("ONNX is required in CI for real protobuf regression fixtures")
            self.skipTest("optional ONNX package unavailable")
        def tensor():
            value = onnx.TensorProto()
            value.name = "external"
            value.data_type = onnx.TensorProto.FLOAT
            value.dims.append(1)
            value.data_location = onnx.TensorProto.EXTERNAL
            entry = value.external_data.add()
            entry.key, entry.value = "location", "not-selected.bin"
            return value
        for location in ("initializer", "attribute", "attributes", "subgraph", "subgraphs",
                         "sparse", "sparse_attribute", "function", "training"):
            with self.subTest(location=location):
                model = onnx.ModelProto()
                graph = model.graph
                if location == "initializer":
                    graph.initializer.add().CopyFrom(tensor())
                elif location == "sparse":
                    graph.sparse_initializer.add().values.CopyFrom(tensor())
                elif location == "function":
                    model.functions.add().node.add().attribute.add().t.CopyFrom(tensor())
                elif location == "training":
                    model.training_info.add().algorithm.initializer.add().CopyFrom(tensor())
                else:
                    attribute = graph.node.add().attribute.add()
                    if location == "attribute":
                        attribute.t.CopyFrom(tensor())
                    elif location == "attributes":
                        attribute.tensors.add().CopyFrom(tensor())
                    elif location == "subgraph":
                        attribute.g.initializer.add().CopyFrom(tensor())
                    elif location == "subgraphs":
                        attribute.graphs.add().initializer.add().CopyFrom(tensor())
                    else:
                        attribute.sparse_tensor.values.CopyFrom(tensor())
                with patch.object(onnx.external_data_helper, "load_external_data_for_model",
                                  side_effect=AssertionError("external files must not be read")):
                    result = inventory.onnx_details(io.BytesIO(model.SerializeToString()), onnx)
                self.assertTrue(result["external_data_present"])
        self.assertFalse(inventory.onnx_details(io.BytesIO(onnx.ModelProto().SerializeToString()),
                                               onnx)["external_data_present"])


if __name__ == "__main__":
    unittest.main()

