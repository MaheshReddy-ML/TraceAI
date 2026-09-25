# Roadmap and implementation status

TraceAI 0.2 implements the local observability foundation and the six engineering workstreams that followed version 0.1. The checkmarks below mean the specified code paths and local fixtures exist; they do **not** mean every optional runtime or external service was exercised in this environment.

- [x] **Evaluation design:** versioned YAML case datasets, separate calibration examples, deterministic rubrics, repeated seeds, case-cluster bootstrap intervals, and controlled reward/specification action environments. The bundled labels are synthetic. Independent labeled data, statistical power, and external validity remain study-specific responsibilities.
- [x] **Incremental checkpoints:** stable-file fingerprinting, sharded-weight completeness checks, a watch session, failed-run resume, same-path model reuse, a single-run project lock, and a maximum local weight-size setting. The watcher refuses changed checkpoints after evaluation; it does not replace an atomic checkpoint writer.
- [x] **Optional inspection:** Transformers next-token logits and per-layer hidden-state norms, plus read-only Hugging Face Trainer metrics. Capability requests are explicit and bounded. Gradients, attention maps, and causal interpretations are not claimed.
- [x] **Bounded agent:** workspace-confined file access, a fixed research command set, and separate opt-ins for model runs and git provenance subprocesses. It does not plan with an LLM or execute arbitrary shell commands.
- [x] **Extension and migration path:** explicitly selected probe/runtime entry points, plugin contract tests, SQLite migration to storage schema 2, and an optional local GGUF runtime alongside MLX, Transformers, Ollama, and mock.
- [x] **Worker and artifact infrastructure:** SQLite job queue, authenticated coordinator, opt-in worker with local model-root confinement, TLS requirement for non-loopback traffic, retry of failed or abandoned jobs, and explicit S3 evidence/report export. Independent experiments can run on separate workers. Checkpoint-level distributed execution and autoscaling are outside this implementation.

## Validation boundary

The test suite covers deterministic fixtures, storage migration, resume/watch behavior, plugin contracts, and an authenticated loopback worker round trip. A cached Transformers causal LM was used for real logits and hidden-state inspection; the prior local MLX and Transformers generation smoke checks are separate from the synthetic study. CUDA hardware, an external TLS worker, a real S3 bucket, Ollama, and GGUF model execution were unavailable for end-to-end validation here. A synthetic calibration score is not a scientific detector validation.

Before citing a model-behavior result, use independently prepared cases and labels, preregister the measurement, review raw evidence, and reproduce on the exact model checkpoints in the target environment. See [methodology](METHODOLOGY.md).
