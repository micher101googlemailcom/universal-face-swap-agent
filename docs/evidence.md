# Evidence register

**Source classes:** [USER] reported directly by project owner in prior conversations; [LOG] error or run observation discussed in prior troubleshooting, without raw logs in this repository; [PLAN] prior proposal/specification, not a measured result; [UNKNOWN] requires local verification. Dates identify conversation context, not current installation state.

| ID | Status | Observation or decision | Limit |
| --- | --- | --- | --- |
| E01 | [USER], 2026-09-27 | VisoMaster Fusion installed at `D:\VMaster_Fusion\VisoMaster-Fusion`. | Version and interface unverified here. |
| E02 | [USER], 2026-05-28 | Eleven source pose categories were reported. Older `GE_NEU_*` IPAdapter and newer `GE_UNI_*` builder `.safetensors` models were described; newer set has nine pose bins plus a master. | Eleven source categories and nine generated bins are different counts; mapping and actual inventory unverified. |
| E03 | [USER], 2026-05-28 | Output identity was judged short of the intended likeness. | No scored comparison or sample in repository. |
| E04 | [USER], 2026-07-26 | Local `model_assets` reportedly holds about 15.9 GB ONNX and 45 `_sm120` TensorRT engines. | Counts, hashes, viability and model compatibility unverified here. Never publish weights or embeddings. |
| E05 | [USER/LOG], 2026-04-22 | `TensorRT EP failed to create context` was seen for FaceLandmark106, later RealEsrganx2Plus; `Input face ... missing from session` followed. Isolated detector/landmark probes appeared successful, but real lazy-build process exited without Python traceback. | Exact root cause unknown; later message can be downstream. |
| E06 | [LOG], 2026-05-17 | VisoMaster loaded models and processed frames; one run was aborted, RAFT unavailable and translation fallback active. Preview reportedly froze on a frame and became blurry. | Queue/feeder diagnosis remains a hypothesis; no definitive trace. |
| E07 | [USER/LOG], 2026-03-13 and 2026-05-28 | ReActor and `facefusion_comfyui` reportedly loaded in ComfyUI. Separate `custom_nodes\facefusion` import failed; another prompt aborted because `reference.png` was invalid at LoadImage. | Loading a node does not establish superior swap quality; the invalid reference image is a separate input error. |
| E08 | [USER], 2026-07-26 and 2026-09-27 | Target is local/offline native ONNX photo/video pipeline. VisoMaster integration stays with the owner and needs an input/output boundary. | No supported VisoMaster automation interface confirmed. |
| E09 | [PLAN] | Use `inswapper_128` with `w600k_r50` as first swapper/embedder candidate and CUDA as baseline. | Compatibility, input tensors, preprocessing, license and quality must be checked against actual files. |
| E10 | [PLAN] | `Video_Start_Robust`, tracking/smoothing/restoration settings and separate hair/mouth/occluder masks were discussed. | Neither preset contents nor benefit validated in this repository. |

Prior material was primarily plans and assistant summaries, not a reproducible experiment log. No winner between VisoMaster, FaceFusion and ReActor, no pose-bin improvement, no performance advantage of TensorRT, and no exact identity threshold has been established.
