# Run CheckMAITE in a container

The CheckMAITE batch container reads one YAML run plan, runs its tasks on one
host, writes durable outputs, and exits. It is intended for local Docker runs,
Kubernetes Jobs, and container-workflow systems. See the
[batch container reference](../reference/container.md) for the complete command,
configuration, security, hardware, and publication contract.

## Build a container

The Dockerfile provides separate CPU and NVIDIA CUDA targets:

```bash
docker build --platform linux/amd64 --target cpu -t checkmaite-batch:cpu .
docker build --platform linux/amd64 --target cuda -t checkmaite-batch:cuda .
```

Choose the target that matches the host. CI builds, scans, and smoke tests
both targets on Linux AMD64. The CUDA target does not make a CPU-only host
provide a GPU.

Both targets run as the non-root user `10001:10001`. The host directories mounted
at `/output` and `/cache` must therefore be writable by that user.

### Measure container size

After building both targets, measure their local uncompressed and compressed
sizes without retaining multi-gigabyte archives:

```bash
python docker/measure_container_sizes.py \
  --json artifacts/container-size-measurement.json \
  checkmaite-batch:cpu checkmaite-batch:cuda
```

The uncompressed value is Docker Engine's image size. The compressed value is a
repeatable gzip level 6 measurement of the `docker image save` stream. It is
useful for comparing builds, but it is not a promise of registry transfer size;
a registry may compress layers differently.

CI enforces stable uncompressed-size ceilings for every supported product:

| Target | Platform | Maximum uncompressed size |
| --- | --- | ---: |
| CPU | Linux AMD64 | 3.00 GiB |
| CUDA | Linux AMD64 | 11.00 GiB |

The exact limits are committed in
[`docker/container-size-budgets.json`](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/docker/container-size-budgets.json).
Image IDs and exact measurements are generated as CI artifacts instead of
being committed because they change whenever a layer changes.

## Prepare the inputs

A typical input directory is:

```text
input/
├── run.yaml
├── data/
│   └── evaluation/
├── models/
│   └── candidate/
└── plugins/
```

[`docker/example-run.yaml`](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/docker/example-run.yaml)
shows an object-detection evaluation using CheckMAITE's COCO dataset loader, ONNX
model, and metric classes. Save a copy as `run.yaml` in the input directory. Its field-level contract is the versioned
[run-plan JSON Schema](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/docker/runtime/schema/run-plan-v1.schema.json).

Object entries use the import path of a Python class or factory function and its
keyword arguments:

```yaml
datasets:
  evaluation:
    class: checkmaite.core.object_detection.dataset_loaders.load_coco_detection_dataset
    args:
      root: data/evaluation/images
      ann_file: data/evaluation/annotations.json
```

Paths in the plan are resolved from the directory containing `run.yaml`.
Constructor arguments may contain another `class` and `args` mapping when one
object needs another object.

## Runtime interface

No environment variables are required. The container accepts these optional
environment variables:

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `CHECKMAITE_CONFIG` | Default for `--config` | `/checkmaite/run.yaml` |
| `CHECKMAITE_OUTPUT_DIR` | Default for `--output` | `/output/results` |
| `CHECKMAITE_CACHE_DIR` | Default for `--cache` and unset library caches | `/cache` |
| `CHECKMAITE_THREADS` | Default for `--threads` | Plan value, then `auto` |
| `CHECKMAITE_DEVICE` | Default for `--device` | Plan value, then `auto` |
| `CHECKMAITE_BATCH_SIZE` | Default for `--batch-size` | Task/capability value |
| `CHECKMAITE_LOG_LEVEL` | Default for `--log-level` | `INFO` |

The expected mounts are:

| Container path | Access | Purpose |
| --- | --- | --- |
| `/checkmaite` | Read-only | Run plan, datasets, models, and trusted plugins |
| `/output` | Writable | Durable reports, analytics, and result manifests |
| `/cache` | Writable | Reusable caches and temporary files |
| `/run/secrets` | Optional, read-only | Secret files consumed by trusted plugins |

The built-in runtime requires no secrets. A trusted plugin that requires a
secret must read it from a file mounted under `/run/secrets`. Do not place
secrets in environment variables, the run plan, or the container. The `/output`
and `/cache` mounts must be writable by UID:GID `10001:10001`.

For every supported setting, a command-line option overrides its environment
variable. Thread and device settings then fall back to `resources` in the run
plan, and `--batch-size` falls back to a task's `config.batch_size` when its
capability supports that field. `HOME` always follows the selected runtime
cache so it remains writable when `--cache` moves away from `/cache`. Existing
`TMPDIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `HF_HOME`, and `TORCH_HOME` values
are preserved. This allows a pre-populated cache, such as a read-only Hugging
Face model mount, to be selected explicitly even when the rest of the runtime
uses `CHECKMAITE_CACHE_DIR`.

A task selects named objects and a capability:

```yaml
tasks:
  - name: baseline
    capability: builtin:MaiteEvaluation
    dataset: evaluation
    model: candidate
    metrics: map50
    config:
      batch_size: 32
```

The short `builtin:` form infers the CheckMAITE problem type from the selected
objects. An explicit form such as `builtin:object_detection.MaiteEvaluation` is
also accepted.

## Run on one host

```bash
mkdir -p output cache

docker run --rm \
  --cpus 8 \
  --memory 32g \
  --mount type=bind,source="$PWD/input",target=/checkmaite,readonly \
  --mount type=bind,source="$PWD/output",target=/output \
  --mount type=bind,source="$PWD/cache",target=/cache \
  checkmaite-batch:cpu \
  run --config /checkmaite/run.yaml --threads auto
```

For a supported NVIDIA host, use the CUDA container and expose the GPU:

```bash
docker run --rm --gpus all \
  --mount type=bind,source="$PWD/input",target=/checkmaite,readonly \
  --mount type=bind,source="$PWD/output",target=/output \
  --mount type=bind,source="$PWD/cache",target=/cache \
  checkmaite-batch:cuda \
  run --config /checkmaite/run.yaml --device auto
```

`--threads auto` uses the CPU affinity and cgroup quota visible to the
container. The runner applies that value to PyTorch and common numerical
libraries. It does not create additional processes or make task execution
concurrent.

The command also supports `--threads N`, `--device auto|cpu|cuda|cuda:N`,
`--batch-size N`, and `--log-level LEVEL`.

## Outputs

A successful run writes:

```text
output/
└── results/
    ├── run-results.json
    ├── analytics/
    └── tasks/
        └── baseline/
            ├── run.json
            └── report.md
```

The exact report filename is selected by the capability. Capabilities that do
not provide a Markdown report still write their run data. Analytics records are
written as Parquet when a run supports analytics extraction. Cache data is kept
under the separately mounted `/cache` directory so it can be reused by later
runs.

Invalid plans exit with status `2`. Execution failures exit with status `1`, and
successful plans exit with status `0`.

## Mounted capability plugins

A trusted Python file can provide a capability without changing the container:

```yaml
tasks:
  - name: project-check
    capability: file:plugins/project_capability.py:ProjectCapability
```

The class must implement the normal CheckMAITE capability interface, and the
objects it receives must follow the relevant MAITE protocols. Mounted Python
files execute with the same permissions as CheckMAITE. Dependencies required by
the plugin must already be installed in the container.
