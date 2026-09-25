# Architecture

TraceAI 0.2 keeps a local experiment engine behind the CLI and Python API:

```text
CLI / Python API ── validated study + dataset
         │
         ▼
Experiment engine ── ModelRuntime (mock, MLX, Transformers, Ollama, GGUF)
         │                    └── optional logits / hidden-state / Trainer-state inspection
         ├── BehaviorProbe or dataset rubric
         ├── SQLite repository ── JSONL evidence ── explicit S3 export
         └── trajectory analyzer ── CLI report / local dashboard

Checkpoint watcher ── stable local checkpoint paths ── engine
Coordinator queue ── authenticated remote workers ── engine on worker
```

## Measurement boundary

The experiment engine scores **outputs**. Optional `inspect` adapters expose bounded summaries of next-token logits, final-token hidden-state norms, or saved Trainer metrics; these inspection results do not change behavioral scores. No adapter claims model intent, live gradients, or causal mechanism. Built-in schema 1 probes are output proxies. Dataset schema 1 contains explicit evaluation and calibration cases; schema 2 experiments can repeat seeds and use controlled task action environments.

One checkpoint's observations and evidence are inserted in one SQLite transaction. Failed runs retain completed checkpoints and can resume with the same configuration. A watcher can append stable checkpoints to its session. SQLite `user_version=2` adds watch and job tables while keeping schema 1 experiments readable. Repository files are owner-only on POSIX where created by TraceAI. A completed experiment also exports `evidence.jsonl`; an archived dataset copy is kept when a dataset is used.

The environment record stores source/config hashes, package versions, platform/hardware, git state, and checkpoint `config.json` hashes. A config hash is **not** a full model-weight hash. Checkpoint watching uses metadata fingerprints and stability intervals; a producer should still publish checkpoints atomically for the strongest guarantee.

## Process and network boundaries

The dashboard binds to `127.0.0.1` and only reads stored reports. It inserts dynamic text through DOM text nodes. The worker coordinator accepts authenticated JSON job requests; non-loopback binding requires an operator-provided TLS certificate and key. A worker must be started explicitly with a model root and secret token. Worker results are validated and imported into the coordinator's SQLite repository. The queue schedules whole experiments, not individual checkpoint tasks. Worker and coordinator share no shell execution protocol. A remote dataset path must exist under the worker's allowed model root, typically on a shared mount.

The agent is a deterministic, permission-bounded command interpreter. It confines configured file reads to a workspace, requires explicit run/subprocess flags before invoking the experiment engine, and has no arbitrary shell action. Its report and evidence commands read stored results.

S3 export is an explicit command for completed artifacts; SQLite stays authoritative. S3 credentials come from the normal `boto3` credential chain and are never stored in the TraceAI configuration.

## Extension contracts

Subclass `ModelRuntime` or `BehaviorProbe` and advertise it with a Python package entry point named `traceai.runtimes` or `traceai.probes`. TraceAI loads a plugin only when its name is selected. Runtime implementations provide `load`, `generate`, `unload`, and `capabilities`; unsupported inspection must raise `UnsupportedCapabilityError`. Probe implementations provide cases and an explicit scoring rule. Treat selected plugins as trusted Python code.

```toml
[project.entry-points."traceai.probes"]
my_probe = "my_package.probes:MyProbe"

[project.entry-points."traceai.runtimes"]
my_runtime = "my_package.runtime:MyRuntime"
```
