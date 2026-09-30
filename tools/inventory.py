"""Offline, read-only inventory of explicitly selected model directories."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat


EXTENSIONS = {".onnx": "onnx", ".safetensors": "safetensors", ".engine": "tensorrt_engine", ".plan": "tensorrt_engine"}


def optional(name):
    try:
        return importlib.import_module(name), None
    except (ImportError, OSError) as exc:
        return None, type(exc).__name__


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def onnx_details(path, module):
    model = module.load(str(path), load_external_data=False)
    def signature(values):
        result = []
        for value in values:
            tensor = value.type.tensor_type
            shape = [dim.dim_param if dim.dim_param else dim.dim_value if dim.HasField("dim_value") else None
                     for dim in tensor.shape.dim] if tensor.HasField("shape") else None
            result.append({"name": value.name, "dtype": module.TensorProto.DataType.Name(tensor.elem_type), "shape": shape})
        return result
    return {"inputs": signature(model.graph.input), "outputs": signature(model.graph.output),
            "external_data_present": any(t.data_location == module.TensorProto.EXTERNAL for t in model.graph.initializer),
            "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import]}


def safetensors_details(path, module):
    with module.safe_open(str(path), framework="np", device="cpu") as handle:
        return {"tensors": [{"key": key, "dtype": handle.get_slice(key).get_dtype(),
                              "shape": list(handle.get_slice(key).get_shape())} for key in handle.keys()]}


def version_hint(root):
    """Read only conventional local release metadata, never execute application code."""
    for name in ("version.txt", "VERSION"):
        candidate = root / name
        if candidate.is_file() and not candidate.is_symlink():
            try:
                value = candidate.read_text(encoding="utf-8")[:128].strip()
                if re.fullmatch(r"[A-Za-z0-9._+ -]{1,80}", value):
                    return {"version": value, "source": name}
            except (OSError, UnicodeError):
                pass
    git = root / ".git"
    if git.is_dir():
        try:
            head = (git / "HEAD").read_text(encoding="ascii").strip()
            if head.startswith("ref: "):
                ref = head[5:]
                if re.fullmatch(r"refs/[A-Za-z0-9/_.-]+", ref) and ".." not in ref:
                    ref_file = git / ref
                    if ref_file.is_file():
                        head = ref_file.read_text(encoding="ascii").strip()
                    else:
                        packed = (git / "packed-refs").read_text(encoding="ascii")
                        head = next((line.split(" ")[0] for line in packed.splitlines() if line.endswith(" " + ref)), "")
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
                return {"revision": head, "source": "git HEAD", "version": None}
        except (OSError, UnicodeError):
            pass
    return {"version": None, "status": "not found in version.txt, VERSION or git HEAD"}


def is_link_or_reparse(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def walk_directory_files(root):
    """Yield directories and regular filenames, propagating all traversal errors."""
    pending = [root]
    while pending:
        current = pending.pop()
        # Recheck queued directories without following Windows junctions, too.
        if is_link_or_reparse(current.lstat()):
            raise OSError("scan directory became a link or reparse point")
        directories, files = [], []
        with os.scandir(current) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                # Unlike os.walk/is_dir, stat does not hide classification errors
                # (including FileNotFoundError). No report is published on failure.
                info = entry.stat(follow_symlinks=False)
                if is_link_or_reparse(info):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    directories.append(current / entry.name)
                elif stat.S_ISREG(info.st_mode):
                    files.append(entry.name)
        yield current, files
        pending.extend(reversed(directories))


def scan(root, label, modules, provenance):
    assets = []
    for current, files in walk_directory_files(root):
        for filename in files:
            path = current / filename
            kind = EXTENSIONS.get(path.suffix.lower())
            if kind is None:
                continue
            entry = {"id": None, "scan_root": label, "type": kind}
            try:
                before = path.lstat()
                if is_link_or_reparse(before):
                    continue
                entry.update({"sha256": digest(path), "size_bytes": before.st_size})
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    entry["warning"] = "source changed during scan; hash may be inconsistent"
                entry["id"] = kind + ":" + entry["sha256"][:16]
                entry["provenance"] = provenance.get(entry["sha256"], {"status": "unknown"})
                if kind == "onnx" and modules["onnx"]:
                    try:
                        entry["onnx"] = onnx_details(path, modules["onnx"])
                    except Exception as exc:
                        entry["metadata_error"] = type(exc).__name__
                elif kind == "safetensors" and modules["safetensors"]:
                    try:
                        entry["safetensors"] = safetensors_details(path, modules["safetensors"])
                    except Exception as exc:
                        entry["metadata_error"] = type(exc).__name__
                elif kind == "tensorrt_engine":
                    entry["engine_hints"] = {"sm120_in_filename": "_sm120" in filename.lower(),
                                             "verified_compatible": False,
                                             "note": "Opaque engine; no deserialization or inference performed"}
            except OSError as exc:
                entry["error"] = type(exc).__name__
            assets.append(entry)
    return assets


def report(data):
    lines = ["# Local model inventory", "", "Paths and filenames omitted; IDs derive from SHA-256.", "",
             "| Kind | Count |", "| --- | ---: |"]
    for kind in sorted(set(EXTENSIONS.values())):
        lines.append(f"| {kind} | {sum(a['type'] == kind for a in data['assets'])} |")
    lines += ["", "## Environment", "", "- ONNX Runtime providers: " + ", ".join(data["providers"] or ["unavailable"]),
              "- Optional modules: " + ", ".join(f"{k}={v}" for k, v in data["dependencies"].items()),
              "- VisoMaster: " + (json.dumps(data["visomaster"], ensure_ascii=False) if data["visomaster"] else "not supplied"),
              "", "## Assets", "", "| ID | Bytes | SHA-256 | Metadata |", "| --- | ---: | --- | --- |"]
    for a in data["assets"]:
        status = "error: " + a["error"] if "error" in a else "metadata error: " + a["metadata_error"] if "metadata_error" in a else "available" if a.get(a["type"], None) else "hash only"
        lines.append(f"| {a['id'] or 'unreadable'} | {a.get('size_bytes', '?')} | {a.get('sha256', '?')} | {status} |")
    lines += ["", "Tensor contracts, pose mapping, engine compatibility and licenses require manual review; filenames are intentionally absent.", ""]
    return "\n".join(lines)


def create_output(path):
    """Reserve a report name without following a late Windows symlink."""
    if os.name != "nt":
        return path.open("x", encoding="utf-8")

    # On Windows, the CRT's exclusive open can create the target of a dangling
    # symlink. CREATE_NEW plus OPEN_REPARSE_POINT checks the directory entry.
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    handle = create_file(str(path), 0x40000000, 0, None, 1, 0x00200080, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except Exception:
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle(handle)
        raise
    try:
        return os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, action="append", required=True, help="Explicit model directory; repeatable")
    parser.add_argument("--visomaster", type=Path, help="Explicit installation directory, version hint only")
    parser.add_argument("--provenance", type=Path, help="Optional JSON mapping full SHA-256 to sanitized provenance objects")
    parser.add_argument("--output", type=Path, required=True, help="Output directory outside all scanned directories")
    args = parser.parse_args(argv)
    roots = [p.resolve(strict=True) for p in args.scan]
    if not all(p.is_dir() for p in roots):
        parser.error("--scan entries must be directories")
    destination = args.output.resolve()
    if any(destination == root or root in destination.parents or destination in root.parents for root in roots):
        parser.error("--output must be disjoint from scan directories")
    if args.visomaster:
        vm = args.visomaster.resolve(strict=True)
        if not vm.is_dir():
            parser.error("--visomaster must be a directory")
    else:
        vm = None
    if vm and (destination == vm or vm in destination.parents or destination in vm.parents):
        parser.error("--output must be disjoint from VisoMaster directory")
    output_json, output_md = destination / "inventory.json", destination / "inventory.md"
    for path in (output_json, output_md):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            parser.error(f"output check failed ({type(exc).__name__})")
        else:
            parser.error("output files already exist; use a fresh output directory")
    provenance = {}
    if args.provenance:
        with args.provenance.open(encoding="utf-8") as stream:
            provenance = json.load(stream)
        if not isinstance(provenance, dict) or any(not re.fullmatch(r"[a-fA-F0-9]{64}", k) or not isinstance(v, dict) for k, v in provenance.items()):
            parser.error("--provenance must map SHA-256 hashes to objects")
        if any(set(v) - {"source", "license", "note"} or
               any(not isinstance(value, str) or len(value) > 200 or "/" in value or "\\" in value or ":" in value
                   for value in v.values()) for v in provenance.values()):
            parser.error("provenance permits only sanitized source, license, note strings without path separators or colons")
        normalized = {key.lower(): value for key, value in provenance.items()}
        if len(normalized) != len(provenance):
            parser.error("--provenance contains duplicate SHA-256 keys differing only in case")
        provenance = normalized
    modules = {}
    dependencies = {}
    for name in ("onnx", "safetensors", "onnxruntime"):
        modules[name], error = optional(name)
        dependencies[name] = "available" if modules[name] else "unavailable: " + error
    providers = []
    if modules["onnxruntime"]:
        try:
            providers = modules["onnxruntime"].get_available_providers()
        except Exception as exc:
            dependencies["onnxruntime"] = "provider query failed: " + type(exc).__name__
    data = {"schema_version": 1, "scan_roots": [f"root-{i+1}" for i in range(len(roots))],
            "dependencies": dependencies, "providers": providers,
            "visomaster": version_hint(vm) if vm else None, "assets": []}
    for index, root in enumerate(roots, 1):
        label = f"root-{index}"
        try:
            data["assets"].extend(scan(root, label, modules, provenance))
        except OSError as exc:
            parser.error(f"{label}: directory traversal failed ({type(exc).__name__}); inventory not written")
    json_text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    markdown_text = report(data)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        # Reserve both names exclusively before writing any inventory data.
        # This also rejects links/files created after the preflight check.
        with create_output(output_json) as json_stream, \
                create_output(output_md) as markdown_stream:
            json_stream.write(json_text)
            markdown_stream.write(markdown_text)
    except OSError as exc:
        parser.error(f"output write failed ({type(exc).__name__}); use a fresh output directory")
    print(f"Inventoried {len(data['assets'])} files in {destination}")


if __name__ == "__main__":
    main()
