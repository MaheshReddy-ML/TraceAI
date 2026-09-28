# Development and validation

Use Python 3.12+ and a local environment:

```bash
uv sync --python 3.12 --extra dev
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests
uv run --extra dev ruff format --check src tests
```

The automated suite uses no cloud credentials or model downloads. It covers the mock engine, versioned dataset rubrics, migration/resume/watch behavior, CLI JSON, plugin contracts, inspection of saved Trainer metrics, agent permissions, S3 adapter calls with a fake client, and a real authenticated loopback coordinator/worker process.

For a local text model, install its extra and use a path returned by `traceai models`:

```bash
uv run --extra transformers traceai probe /absolute/local/snapshot \
  --runtime transformers --behavior baseline --device cpu
uv run --extra transformers traceai inspect /absolute/local/snapshot \
  --capability logits --prompt 'What is 2 + 2?' --json
```

The Transformers adapter uses CUDA automatically when PyTorch reports it available, or `device: cuda` can require it. CPU/CUDA placement and particular model architectures should be tested on the target machine. MLX requires Apple Silicon and the `mlx` extra. GGUF requires the `llama_cpp` extra and a local `.gguf` file. No adapter pulls weights.

## Evaluate your own training checkpoints

Run `traceai guide` for the local workflow. `traceai dataset template --output cases.yaml` writes a synthetic YAML example; replace its arithmetic cases and calibration outputs with your own held-out task cases. Then use `traceai init --guided --path study.yaml` in a terminal to create a study. The setup wizard lists discovered model paths, accepts distinct checkpoint IDs and paths, and can attach your edited cases YAML. It validates the setup without loading weights. For automation, use `traceai init --runtime transformers --model /path/to/model --checkpoint before=/path/to/first --checkpoint after=/path/to/second --dataset /path/to/cases.yaml --path study.yaml`.

Validate the generated file with `traceai config validate study.yaml`, calibrate labeled rubrics with `traceai dataset calibrate /path/to/cases.yaml`, then run the study. After the run, `traceai guide EXPERIMENT_ID` selects a descriptive change to investigate and gives an exact evidence command. Review the raw prompts and responses, adjust your training hypothesis, and compare another checkpoint on the same held-out cases. TraceAI evaluates checkpoints; your training pipeline owns weight updates. Built-in probes remain narrow output proxies, and a synthetic or small dataset cannot establish general model behavior.

## Worker setup

First validate local reproducibility. Then use a shared model/dataset mount on each trusted worker. Set the same long random secret in the coordinator and worker environments; do not put it on a command line:

```bash
export TRACEAI_WORKER_TOKEN="<at-least-24-random-characters>"
traceai --project .traceai jobs submit study.yaml
traceai --project .traceai worker serve --host 127.0.0.1 --port 8766
# In another terminal or trusted host:
traceai worker run --server http://127.0.0.1:8766 \
  --workdir /absolute/worker-runs --model-root /absolute/shared-models
traceai --project .traceai jobs list
```

For a non-loopback coordinator, supply `--cert` and `--key`, use an `https://` worker URL, and optionally `--ca-file` for a private CA. The worker rejects model and dataset paths outside `--model-root`. Jobs are whole experiments; a failed or abandoned job can be explicitly requeued with `traceai jobs retry ID` after confirming its previous worker stopped. The coordinator's JSON result limit is 16 MiB; larger studies should be split or the transport extended.

## Artifact export

Install `.[s3]`, configure standard AWS credentials, then run `traceai artifacts upload ID --destination s3://BUCKET/PREFIX`. The command uploads a JSON report, JSONL evidence, and the archived dataset when present. It does not alter the authoritative local SQLite data. Confirm IAM permissions and bucket policy in the target account before relying on a live export.

New probes should document exact case text, scoring rule, version, calibration, and limits. New runtimes should advertise capabilities, load only the selected local resource, and release it in `unload`. See [architecture](ARCHITECTURE.md) and [methodology](docs/METHODOLOGY.md).
