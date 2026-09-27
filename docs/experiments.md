# Next experiments

Use the [run record](workflow.md#suggested-run-record) for each case. These are proposals, **not completed results**. Order minimizes ambiguity and preserves the existing local installation.

| Order | Experiment and controlled variants | Evidence to save | Decision |
| --- | --- | --- | --- |
| 1 | Inventory actual ONNX models, `.safetensors` sets, `_sm120` engines and installed VisoMaster version; verify model pair tensor contract and pose mapping without modifying source files | Hashes, tensor specs, provenance, provider availability, inventory with private paths redacted | Pick a compatible baseline pair; quarantine unknown assets conceptually |
| 2 | Small CUDA-only still baseline: same source/targets with frontal, profile, expression and occlusion; no restorer versus one restorer | Outputs, alignment crops, run settings, identity/artifact review | Identify baseline and failure classes |
| 3 | Fixed short video: same baseline on simple motion, occlusion, multiple faces; compare tracking on/off or selection policies one variable at a time | Frame count, face ID trace, flicker/blur notes, FPS, VRAM | Establish temporal gate and wrong-target rate |
| 4 | Reference choice: one reference vs multiple verified poses; master vs mapped `GE_UNI_*` only if compatible loader exists; include older `GE_NEU_*` only through its verified loader | Per-pose blinded identity/edge ratings and failure count | Keep pose routing only if it beats baseline without extra wrong faces |
| 5 | Owner-operated VisoMaster and isolated FaceFusion/ReActor comparison on identical frozen cases where available | Version/settings, standardized output crops, time/VRAM | Decide whether optional adapter adds measurable value |
| 6 | TensorRT on one validated ONNX stage, then chain: isolated deserialize/inference, shape coverage, CUDA comparison, repeated full runs | Provider logs, numeric outputs, quality review, speed/VRAM/crash rate | Enable only on verified improvement with no gate regression |

No experiment requires cloud inference or upload of source identities. VisoMaster integration implementation belongs to the owner; a future adapter only needs an agreed manifest and input/output contract after a real interface has been verified.
