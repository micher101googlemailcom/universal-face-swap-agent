# Universal Face Swap Auto-Correction Agent

Research, workflow and quality documentation for an **offline, single-machine** photo/video face-swap project. The intended implementation is a native ONNX face pipeline using existing local models; this repository currently contains documentation, not a working agent. Diffusion and VR are later scope.

Start with [workflow](docs/workflow.md), [evidence register](docs/evidence.md), [quality gates](docs/quality-gates.md), [troubleshooting](docs/troubleshooting.md), and [next experiments](docs/experiments.md).

Experiment 1 has a [local read-only inventory tool](docs/inventory.md). It records metadata only for explicitly selected model directories and produces JSON and Markdown for private review.

## Boundaries

- Keep original footage, reference images, model weights and private face embeddings off the public repository. Record hashes and sanitized metadata instead.
- VisoMaster Fusion is an existing local tool and optional read-only comparison path. Its integration will be handled separately by the owner; do not assume a CLI/API or modify its installation.
- FaceFusion/ReActor are comparison candidates, not mandatory production dependencies.
- ONNX Runtime CUDA is the proposed reproducible baseline. Treat cached TensorRT engines as unverified until tested end to end.
- Only process material with the rights and consent needed for its intended use; mark synthetic output when distributed.
