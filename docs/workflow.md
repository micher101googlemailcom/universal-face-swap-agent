# Working workflow and boundaries

## Proposed reproducible path

1. **Inventory locally:** record application versions, device, providers, model filenames and SHA-256 hashes, input/output tensor specs, source rights and test clip properties. Do not copy personal source photos, embeddings or weights into Git.
2. **Fixed test set:** choose one authorized source identity and target scenes covering frontal, profile, occlusion, expression, motion and multiple faces. Freeze input frames/timecodes so every run is comparable.
3. **CUDA baseline:** detect target face → track/select intended face in video → align/crop → obtain source embedding from candidate reference or pose model → swap with compatible ONNX pair → composite into original frame → encode output. Confirm each step independently. Begin with proposed `inswapper_128` / `w600k_r50` only after verifying actual contracts. Record failures as well as successful frames.
4. **Quality review:** apply [quality gates](quality-gates.md) to stills and short video; compare against an unmodified target and a fixed baseline. Keep restoration, masks and color correction separately toggleable so each effect can be attributed.
5. **Optional comparison:** export the same sanitized test manifest to owner-operated VisoMaster Fusion or isolated FaceFusion/ReActor tests. Compare outputs on identical inputs; these tools are not production runtime requirements.
6. **Acceleration:** only after CUDA correctness and repeatability, test one TensorRT engine at a time following [troubleshooting](troubleshooting.md).

## Pose assets

The reported **11 source pose categories** and **nine `GE_UNI_*` bins plus master** need an explicit owner-verified mapping; do not infer correspondence by index. The `GE_NEU_*` family is an older IPAdapter set, not an interchangeable ONNX embedding. Treat `.safetensors` as format evidence only: verify which loader created it, expected tensor keys, reference selection logic, pose boundaries, and whether the chosen pipeline can consume it. If it cannot, test reference photos/embeddings first and retain the pose assets for an appropriate optional adapter.

## Suggested run record

| Field | Record |
| --- | --- |
| Dataset/case | Synthetic case ID, source/target hashes, frame/time range, rights/consent status |
| Environment | OS, GPU, driver, runtime/provider versions, VRAM limit |
| Models | Name, SHA-256, provenance/license, input/output shape, alignment and normalization |
| Settings | Detector, tracker, selected face, pose bin/master/reference, mask, restorer, seed if applicable |
| Result | Success/failure, runtime, peak VRAM, identity assessment, artifacts, missed/wrong faces, flicker |
| Comparison | Baseline run ID, reviewer notes, links to local protected outputs |

Keep run records free of biometric data and machine-specific secrets. No numeric settings in this document are asserted as proven optimal.
