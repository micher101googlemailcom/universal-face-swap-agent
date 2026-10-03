# Experiment 1: local model inventory

Run with Python 3.10+ on the machine containing your models. No network client is used. The tool reads only selected directories and optional local metadata; it neither starts inference nor deserializes TensorRT engines. Beneath each resolved scan root, symlinked files/directories and all Windows reparse points (including NTFS junctions and mount points) are excluded. Output is refused inside or above a scanned directory or the VisoMaster directory; existing output files are never overwritten.

```powershell
# Windows PowerShell 5.1; adjust paths to existing directories on your PC.
python tools/inventory.py --scan "C:\Models" --scan "D:\ModelAssets" --visomaster "D:\VMaster_Fusion\VisoMaster-Fusion" --output "C:\inventory-review-01"
```

The output directory contains `inventory.json` (structured detail) and `inventory.md` (short summary). Paths and basenames are omitted entirely; `root-1`, `root-2` correspond to the order of `--scan` arguments. Asset IDs use the first 16 hex digits of SHA-256; the full hash is included. Duplicate content may share an ID; use the scan root and full hash to distinguish records. The report may contain tensor names/keys and manually entered provenance: inspect both files before sharing. Keep generated inventories, model weights, personal images and embeddings out of Git.

With no third-party packages, the tool still hashes files, records byte sizes and extension-based type, and identifies `_sm120` in engine basenames without publishing the basename. Install optional packages **in the Python environment used to run the tool** for richer metadata:

```powershell
python -m pip install onnx safetensors onnxruntime-gpu
```

`onnx` reads input/output names, element types, symbolic/static shapes, opsets and external-data presence without loading external weight files. `safetensors` metadata is read from the bounded JSON header on the same verified file stream used for hashing, without materializing tensors or reopening the asset path. Header shape, dtype and data ranges are validated; unsupported dtypes produce a metadata error rather than guessed metadata. `onnxruntime` reports locally available execution providers; this is only a provider listing, not proof of working CUDA/TensorRT inference. Engine filename hints are not evidence of compatibility. Missing libraries and per-file parse failures are recorded and scanning continues. ONNX models with external weights have an incomplete hash chain until their referenced files are inventoried separately; external files are not traversed automatically.

If a selected root or nested directory cannot be enumerated, or an entry cannot be classified with a non-following stat call, the command exits with code 2 before writing either report. This includes entries disappearing during classification. The error identifies only the scan-root label and exception type, never the directory path or raw exception message. An incomplete traversal is not published as a successful inventory.

Both report names are checked without following links, so dangling output symlinks are also rejected. Both files are then opened for exclusive creation before any inventory data is written, preventing overwrites or following links that appear after the check. An output failure exits with code 2 and may leave empty or incomplete report files; use a fresh output directory after correcting the cause. Existing entries are never deleted as cleanup.

For known provenance, provide a local JSON file keyed by the **full** SHA-256, with only short sanitized `source`, `license`, and `note` strings. Example:

```json
{"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef": {"source": "owner supplied", "license": "review pending", "note": "candidate embedder"}}
```

Pass it using `--provenance C:\private\provenance.json`. Unknown entries remain `{"status":"unknown"}`. Only the existence of a standard `version.txt`, `VERSION`, or Git HEAD in `--visomaster` is inspected; if none is present, version remains unknown. A Git revision is not necessarily a release version. Manually review the reported version and asset contracts before selecting the baseline pair or mapping pose bins.

Provenance hash keys accept uppercase, lowercase, or mixed-case hexadecimal and are normalized to lowercase after validation. Exact duplicate JSON keys and hash keys that differ only in case are rejected to prevent silent overwrites; the existing provenance sanitization rules still apply.

Run the regression tests from the repository root with `python -m unittest discover -s tests -v`. The base suite uses the standard library, temporary synthetic files and simulated filesystem errors. A real ONNX protobuf regression also checks initializer, attribute, nested graph, sparse, function and training tensors; it is skipped locally if ONNX is absent. CI installs ONNX and requires this test on Linux and Windows; no model downloads or inference are involved. Reparse-point filtering is tested on all platforms; an additional native junction/cycle test runs only on Windows. Symlink tests require permission to create symlinks and report a skip if that capability is unavailable.


Directory traversal and report creation stay anchored to open directories: POSIX uses descriptor-relative no-follow operations; Windows pins every ancestor with no-follow handles that deny write/delete sharing. Late links are rejected. A renamed POSIX output directory can receive reports through its original pinned handle, but replacement links are never followed. Windows sharing conflicts fail closed. Hashing and optional metadata use one verified regular-file stream. Git metadata components, including `.git`, `HEAD` and refs, use the same no-follow access.

The final source-change check runs after hashing and all metadata reads, including failed metadata extraction. Detected identity, size or timestamp changes add a warning that the hash, metadata and provenance may be inconsistent. This check is not an atomic filesystem snapshot. Malformed Git ref components leave the optional version hint unknown instead of aborting the inventory. ONNX interface contracts other than dense tensors produce `UnsupportedONNXTypeError`; their types are not misrepresented as `UNDEFINED` tensors.
