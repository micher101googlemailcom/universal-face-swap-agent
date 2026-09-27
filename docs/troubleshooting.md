# Troubleshooting: reproduce before changing settings

## TensorRT context failure

Observed previously: `TensorRT EP failed to create context` for FaceLandmark106 and RealEsrganx2Plus, with `Input face ... missing from session` later. Root cause is **not established**; the latter may follow a failed upstream stage.

1. Capture exact model hash, provider order/options, GPU/driver/runtime versions, engine cache path, shape, batch, FP16 flags, available/peak VRAM and complete stderr/exit code.
2. Run the same **real inference input** through ONNX Runtime CUDA. Confirm outputs and compare tolerances appropriate to each model. A successful isolated load/cache build is insufficient.
3. Reproduce the failing TensorRT model alone in a fresh process, then in the full chain. Verify engine deserialization, profile/input shapes, plugins, precision, device compatibility and output sanity. Cache filenames containing `sm120` are not proof of compatibility.
4. Preserve existing engines. Move a suspect cache to an isolated test directory only with a copy, then rebuild/test there. Avoid global package or driver upgrades as a first response.
5. If the process exits without traceback, capture Windows Event Viewer entry, process exit code, GPU memory and provider logs. Compare CUDA-only on the same input to separate stage-specific failures from resource contention.
6. Keep CUDA as fallback until TensorRT passes load, inference, quality comparison, repeated runs and measured end-to-end benefit.

## Other known failure classes

| Symptom | Check first | Do not conclude |
| --- | --- | --- |
| `Input face ... missing from session` | Earliest failing upstream detector/landmark/provider step and selected face ID | That the input image itself is necessarily absent |
| ComfyUI `LoadImage` rejects `reference.png` | Path, file validity, image decode and workflow node input | That ReActor or GPU memory caused the prompt abort |
| Preview freezes or gets blurrier over time | Frame index/time stamps at decode, processing, display and output; queue sizes; compare CUDA-only minimal graph | That RAFT fallback or TensorRT is proven responsible |
| Wrong face / identity changes | Tracker trace, face-selection policy, source reference and bin mapping | That restoration alone will fix selection |
| Low likeness but stable tracking | Compare source reference, embedding/model pairing, crop/alignment, then pose-bin variants | That higher resolution or extra restoration guarantees identity |

Record the smallest failing case before changing more than one variable.
