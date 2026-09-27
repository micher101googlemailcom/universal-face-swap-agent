# Experiment 1: local model inventory

Run with Python 3.10+ on the machine containing your models. No network client is used. The tool reads only selected directories and optional local metadata; it neither starts inference nor deserializes TensorRT engines. It follows no symlinked files/directories. Output is refused inside or above a scanned directory or the VisoMaster directory; existing output files are never overwritten.

```powershell
# Windows PowerShell 5.1; adjust paths to existing directories on your PC.
python tools/inventory.py --scan "C:\Models" --scan "D:\ModelAssets" --visomaster "D:\VMaster_Fusion\VisoMaster-Fusion" --output "C:\inventory-review-01"
```

The output directory contains `inventory.json` (structured detail) and `inventory.md` (short summary). Paths and basenames are omitted entirely; `root-1`, `root-2` correspond to the order of `--scan` arguments. Asset IDs use the first 16 hex digits of SHA-256; the full hash is included. Duplicate content may share an ID; use the scan root and full hash to distinguish records. The report may contain tensor names/keys and manually entered provenance: inspect both files before sharing. Keep generated inventories, model weights, personal images and embeddings out of Git.

With no third-party packages, the tool still hashes files, records byte sizes and extension-based type, and identifies `_sm120` in engine basenames without publishing the basename. Install optional packages **in the Python environment used to run the tool** for richer metadata:

```powershell
python -m pip install onnx safetensors onnxruntime-gpu
```

`onnx` reads input/output names, element types, symbolic/static shapes, opsets and external-data presence without loading external weight files. `safetensors` reads keys, dtypes and shapes through slice metadata without materializing tensors. `onnxruntime` reports locally available execution providers; this is only a provider listing, not proof of working CUDA/TensorRT inference. Engine filename hints are not evidence of compatibility. Missing libraries and per-file parse failures are recorded and scanning continues. ONNX models with external weights have an incomplete hash chain until their referenced files are inventoried separately; external files are not traversed automatically.

For known provenance, provide a local JSON file keyed by the **full** SHA-256, with only short sanitized `source`, `license`, and `note` strings. Example:

```json
{"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef": {"source": "owner supplied", "license": "review pending", "note": "candidate embedder"}}
```

Pass it using `--provenance C:\private\provenance.json`. Unknown entries remain `{"status":"unknown"}`. Only the existence of a standard `version.txt`, `VERSION`, or Git HEAD in `--visomaster` is inspected; if none is present, version remains unknown. A Git revision is not necessarily a release version. Manually review the reported version and asset contracts before selecting the baseline pair or mapping pose bins.
