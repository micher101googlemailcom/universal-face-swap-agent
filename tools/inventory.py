"""Offline, read-only inventory of explicitly selected model directories."""

import argparse
from contextlib import ExitStack
import struct
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat

if __package__:
    from . import secure_io
else:
    import secure_io


EXTENSIONS = {".onnx": "onnx", ".safetensors": "safetensors", ".engine": "tensorrt_engine", ".plan": "tensorrt_engine"}


def optional(name):
    try:
        return importlib.import_module(name), None
    except (ImportError, OSError) as exc:
        return None, type(exc).__name__


def digest(stream):
    h = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def external_tensors(message, module):
    """Walk protobuf message fields, including subgraphs, sparse tensors/functions."""
    if isinstance(message, module.TensorProto):
        return message.data_location == module.TensorProto.EXTERNAL or bool(message.external_data)
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            repeated = field.is_repeated if hasattr(field, "is_repeated") else field.label == field.LABEL_REPEATED
            values = value if repeated else (value,)
            if any(external_tensors(child, module) for child in values):
                return True
    return False


class UnsupportedONNXTypeError(ValueError):
    """The report schema only represents dense tensor interface contracts."""


def onnx_details(stream, module):
    stream.seek(0)
    # fdopen streams have an integer name; do not let ONNX infer a path/format.
    model = module.load(stream, format="protobuf", load_external_data=False)
    def signature(values):
        result = []
        for value in values:
            if value.type.WhichOneof("value") != "tensor_type":
                raise UnsupportedONNXTypeError("unsupported ONNX interface type")
            tensor = value.type.tensor_type
            shape = [dim.dim_param if dim.dim_param else dim.dim_value if dim.HasField("dim_value") else None
                     for dim in tensor.shape.dim] if tensor.HasField("shape") else None
            result.append({"name": value.name, "dtype": module.TensorProto.DataType.Name(tensor.elem_type), "shape": shape})
        return result
    return {"inputs": signature(model.graph.input), "outputs": signature(model.graph.output),
            "external_data_present": external_tensors(model, module),
            "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import]}


def safetensors_details(stream, module):
    # safe_open only accepts paths. Read the bounded JSON header from the same
    # verified stream instead of reopening a mutable path or loading GPU tensors.
    stream.seek(0)
    length = struct.unpack("<Q", stream.read(8))[0]
    if length > 100_000_000 or length > os.fstat(stream.fileno()).st_size - 8:
        raise ValueError("invalid safetensors header length")
    raw = stream.read(length)
    if not raw.startswith(b"{"):
        raise ValueError("invalid safetensors header")
    header = json.loads(raw, object_pairs_hook=unique_object)
    bits = {"BOOL": 8, "U8": 8, "I8": 8, "I16": 16, "U16": 16, "F16": 16,
            "BF16": 16, "I32": 32, "U32": 32, "F32": 32, "I64": 64, "U64": 64,
            "F64": 64, "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8}
    tensors, ranges = [], []
    for key, value in header.items():
        if key == "__metadata__":
            if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
                raise ValueError("invalid safetensors metadata")
            continue
        dtype, shape, offsets = value["dtype"], value["shape"], value["data_offsets"]
        if dtype not in bits or not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("unsupported dtype or invalid shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(d) is not int or d < 0 for d in offsets):
            raise ValueError("invalid offsets")
        size = bits[dtype]
        for dim in shape:
            size *= dim
        start, end = offsets
        if size % 8 or end < start or end - start != size // 8:
            raise ValueError("tensor size does not match offsets")
        ranges.append((start, end))
        tensors.append({"key": key, "dtype": dtype, "shape": shape})
    offset = 0
    for start, end in sorted(ranges):
        if start != offset:
            raise ValueError("overlapping tensors or unindexed data")
        offset = end
    if offset != os.fstat(stream.fileno()).st_size - 8 - length:
        raise ValueError("invalid safetensors data length")
    return {"tensors": sorted(tensors, key=lambda item: item["key"])}



def version_hint(root):
    """Read conventional metadata through pinned no-follow directory handles."""
    try:
        with secure_io.anchor(root) as (directory, _):
            for name in ("version.txt", "VERSION"):
                try:
                    with directory.file(name) as stream:
                        value = stream.read(512).decode("utf-8")[:128].strip()
                    if re.fullmatch(r"[A-Za-z0-9._+ -]{1,80}", value):
                        return {"version": value, "source": name}
                except (OSError, UnicodeError):
                    pass
            with directory.child(".git") as git:
                with git.file("HEAD") as stream:
                    head = stream.read(512).decode("ascii").strip()
                if head.startswith("ref: "):
                    ref = head[5:]
                    parts = ref.split("/")
                    if (re.fullmatch(r"refs/[A-Za-z0-9/_.-]+", ref) and ".." not in ref
                            and all(part not in ("", ".", "..") for part in parts)):
                        try:
                            with ExitStack() as stack:
                                parent = git
                                for part in parts[:-1]:
                                    parent = stack.enter_context(parent.child(part))
                                with parent.file(parts[-1]) as stream:
                                    head = stream.read(128).decode("ascii").strip()
                        except FileNotFoundError:
                            with git.file("packed-refs") as stream:
                                head = next((line.decode("ascii").split(" ")[0] for line in stream
                                             if line.decode("ascii").strip().endswith(" " + ref)), "")
                if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
                    return {"revision": head, "source": "git HEAD", "version": None}
    except (OSError, UnicodeError):
        pass
    return {"version": None, "status": "not found in version.txt, VERSION or git HEAD"}


def is_link_or_reparse(info):
    return secure_io.linked(info)


def walk_directory_files(root):
    """Keep parent handles live until enumeration and file reads have finished."""
    def walk(directory):
        directories, files = [], []
        with directory.entries() as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                info = entry.stat(follow_symlinks=False)
                if is_link_or_reparse(info):
                    continue
                if stat.S_ISDIR(info.st_mode):
                    directories.append(entry.name)
                elif stat.S_ISREG(info.st_mode):
                    files.append(entry.name)
        yield directory, files
        for name in directories:
            with directory.child(name) as child:
                yield from walk(child)
    with secure_io.anchor(root) as (directory, _):
        yield from walk(directory)


def scan(root, label, modules, provenance):
    assets = []
    for current, files in walk_directory_files(root):
        for filename in files:
            path = current.path / filename
            kind = EXTENSIONS.get(path.suffix.lower())
            if kind is None:
                continue
            entry = {"id": None, "scan_root": label, "type": kind}
            try:
                with current.file(filename) as stream:
                    before = os.fstat(stream.fileno())
                    entry.update({"sha256": digest(stream), "size_bytes": before.st_size})
                    entry["id"] = kind + ":" + entry["sha256"][:16]
                    entry["provenance"] = provenance.get(entry["sha256"], {"status": "unknown"})
                    if kind == "onnx" and modules["onnx"]:
                        try:
                            entry["onnx"] = onnx_details(stream, modules["onnx"])
                        except Exception as exc:
                            entry["metadata_error"] = type(exc).__name__
                    elif kind == "safetensors" and modules["safetensors"]:
                        try:
                            entry["safetensors"] = safetensors_details(stream, modules["safetensors"])
                        except Exception as exc:
                            entry["metadata_error"] = type(exc).__name__
                    elif kind == "tensorrt_engine":
                        entry["engine_hints"] = {"sm120_in_filename": "_sm120" in filename.lower(),
                                                 "verified_compatible": False,
                                                 "note": "Opaque engine; no deserialization or inference performed"}
                    after = os.fstat(stream.fileno())
                    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                    if any(getattr(before, field) != getattr(after, field) for field in fields):
                        entry["warning"] = "source changed during scan; hash, metadata and provenance may be inconsistent"
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


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def main(argv=None):
    with ExitStack() as stack:
        return run(argv, stack)


def run(argv, stack):
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
    try:
        output_anchor = stack.enter_context(secure_io.anchor(destination, allow_missing=True))
    except OSError as exc:
        parser.error(f"output check failed ({type(exc).__name__})")
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
            try:
                provenance = json.load(stream, object_pairs_hook=unique_object)
            except ValueError:
                parser.error("--provenance contains duplicate JSON keys or invalid JSON")
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
        with secure_io.materialize(*output_anchor) as directory:
            # Reserve both names before writing any inventory data.
            with directory.file("inventory.json", create=True) as json_stream, \
                    directory.file("inventory.md", create=True) as markdown_stream:
                json_stream.write(json_text)
                markdown_stream.write(markdown_text)
    except OSError as exc:
        parser.error(f"output write failed ({type(exc).__name__}); use a fresh output directory")
    print(f"Inventoried {len(data['assets'])} files in {destination}")


if __name__ == "__main__":
    main()
