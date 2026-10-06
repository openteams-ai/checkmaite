# Batch container reference

The `checkmaite-container` command executes one finite version 1 YAML run plan
on one host, writes durable results, and exits. It is the entry point of the
CheckMAITE batch container. It is a command-line program, not a long-running
service, so it has no HTTP interface or health endpoint.

The generated [run-plan JSON Schema](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/docker/runtime/schema/run-plan-v1.schema.json)
is the machine-readable form of the plan contract described here. Editors with
YAML language-server support can validate and complete a plan that starts with:

```yaml
# yaml-language-server: $schema=<path or URL to run-plan-v1.schema.json>
```

## Product variants and platforms

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Variant | Build target | Supported platform | Accelerator packages |
| --- | --- | --- | --- |
| CPU | `cpu` | Linux AMD64 | CPU-only PyTorch and ONNX Runtime |
| NVIDIA CUDA | `cuda` | Linux AMD64 | CUDA 13 PyTorch, ONNX Runtime GPU, cuDNN, NCCL, and Triton |

<!-- markdownlint-enable MD013 -->

The CUDA container requires a compatible NVIDIA GPU, host driver, NVIDIA
Container Toolkit, and a runtime that exposes the device. Installing or running
the CUDA container does not add a GPU to a CPU-only host. The CPU container does
not contain CUDA, NVIDIA, or Triton Python packages.

The CPU container is built on the Docker Official `ubuntu:24.04` image. The CUDA
container is built on NVIDIA's CUDA 13 cuDNN runtime image for Ubuntu 24.04.
Both are digest-pinned, and every operating-system package is fixed to one
Ubuntu archive snapshot. Neither image contains a package manager.

Both variants run as UID:GID `10001:10001`. The application environment under
`/opt/venv` and the working directory `/checkmaite` are root-owned and are not
writable by that account.

## Run the command from a checkout

The command lives in an unpublished package under `docker/runtime`, which has
its own lock with CPU and CUDA PyTorch builds. From a CheckMAITE checkout with
[uv](https://docs.astral.sh/uv/) installed, run it with the `cpu` extra (or
`cuda` on Linux AMD64 with an NVIDIA GPU):

```bash
uv run --project docker/runtime --extra cpu checkmaite-container --help
```

The first call creates `docker/runtime/.venv`. To run the sample plan,
[`docker/example-run.yaml`](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/blob/main/docker/example-run.yaml),
copy it into a directory laid out like this:

```text
input/
├── run.yaml
├── data/evaluation/
│   ├── annotations.json      # COCO annotations
│   └── images/
└── models/candidate/
    ├── model.onnx
    └── config.json           # ONNX model metadata
```

and run it, choosing writable output and cache directories:

```bash
uv run --project docker/runtime --extra cpu checkmaite-container run \
  --config input/run.yaml --output output --cache cache
```

Relative paths inside the plan are resolved from the directory that contains
the plan, so the same plan runs unchanged from a checkout or from a container
mount.

## Command line

Running the command with no arguments is equivalent to:

```text
checkmaite-container run --config /checkmaite/run.yaml
```

The complete syntax is:

```text
checkmaite-container run [--config PLAN] [--output DIR] [--cache DIR]
                         [--secrets DIR]
                         [--threads auto|N]
                         [--device auto|cpu|cuda|cuda:N]
                         [--batch-size N] [--log-level LEVEL]
```

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Option | Default | Meaning |
| --- | --- | --- |
| `--config PLAN` | `CHECKMAITE_CONFIG`, then `/checkmaite/run.yaml` | Version 1 YAML run plan |
| `--output DIR` | `CHECKMAITE_OUTPUT_DIR`, then `/output/results` | Durable output directory |
| `--cache DIR` | `CHECKMAITE_CACHE_DIR`, then `/cache` | Reusable cache and temporary-data directory |
| `--secrets DIR` | `CHECKMAITE_SECRETS_DIR`, then `/run/secrets` | Directory of secret files for trusted plugins; it need not exist |
| `--threads auto\|N` | `CHECKMAITE_THREADS`, plan `resources.threads`, then `auto` | Process thread budget; `N` must be positive and is capped at visible CPUs |
| `--device auto\|cpu\|cuda\|cuda:N` | `CHECKMAITE_DEVICE`, plan `resources.device`, then `auto` | Execution device |
| `--batch-size N` | `CHECKMAITE_BATCH_SIZE`, each task's `config.batch_size`, then capability default | Positive override for every task whose capability has a `batch_size` field |
| `--log-level LEVEL` | `CHECKMAITE_LOG_LEVEL`, then `INFO` | `DEBUG`, `INFO`, `WARNING` (or `WARN`), `ERROR`, or `CRITICAL` |
| `--version` | - | Print the CheckMAITE version and exit |
| `--help`, `-h` | - | Print the operational interface and exit |

<!-- markdownlint-enable MD013 -->

Every option is optional. An option beginning with `-` is treated as an option
to the default `run` command. For example, passing only `--threads 2` runs the
default plan with two threads.

## Environment variables

No environment variable is required. These variables configure the command:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHECKMAITE_CONFIG` | `/checkmaite/run.yaml` | Default for `--config` |
| `CHECKMAITE_OUTPUT_DIR` | `/output/results` | Default for `--output` |
| `CHECKMAITE_CACHE_DIR` | `/cache` | Default for `--cache` |
| `CHECKMAITE_SECRETS_DIR` | `/run/secrets` | Default for `--secrets` |
| `CHECKMAITE_THREADS` | Plan value, then `auto` | Default for `--threads` |
| `CHECKMAITE_DEVICE` | Plan value, then `auto` | Default for `--device` |
| `CHECKMAITE_BATCH_SIZE` | Task/capability value | Default for `--batch-size` |
| `CHECKMAITE_LOG_LEVEL` | `INFO` | Default for `--log-level` |

While the plan runs, the command sets `CHECKMAITE_SECRETS_DIR` to the selected
secrets directory, and `CHECKMAITE_THREADS` and `CHECKMAITE_DEVICE` to the
resolved thread count and device, so trusted capabilities and plugins can read
them.

The selected cache directory always sets `HOME`, so it stays writable when
`--cache` moves away from `/cache`. It also provides defaults for `TMPDIR`,
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `HF_HOME`, and `TORCH_HOME`. Existing values
for those five variables are kept, so a deployment can point them at a
separately mounted, pre-populated library cache. The CUDA variant inherits
NVIDIA's runtime variables and driver compatibility constraints directly from
its pinned NVIDIA base.

## Input precedence

When a setting is available from more than one source, precedence is:

1. command-line option;
2. environment variable;
3. run-plan value, where the setting has a plan field;
4. built-in default.

So `--threads` and `--device` override `resources` in the plan, and
`--batch-size` overrides a task's `config.batch_size` when the capability
supports that field.

## Directories and secrets

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Default path | Access | Purpose |
| --- | --- | --- |
| `/checkmaite` | Read-only | Run plan, datasets, models, and trusted plugin files |
| `/output` | Writable | Result summary, task runs, reports, and analytics |
| `/cache` | Writable | CheckMAITE caches, library caches, and temporary files |
| `/run/secrets` | Read-only | Secret files, only for plugins that need them |

<!-- markdownlint-enable MD013 -->

The built-in runtime requires no secrets. A trusted plugin that needs a
password, token, key, or certificate must read it from a file in the directory
named by `CHECKMAITE_SECRETS_DIR`. Do not put secrets in environment variables,
the run plan, command-line arguments, or scheduler metadata.

In the container, the conventional input layout places `run.yaml` at the root of
`/checkmaite`, datasets under `/checkmaite/data/`, and models under
`/checkmaite/models/`. Results default to `/output/results/`. Every path remains
explicit in the plan or command line when a deployment needs a different
mounted layout.

The output and cache mounts must be writable by `10001:10001`. Large datasets,
models, reports, and analytics belong in mounted storage, not in environment
variables, ConfigMaps, database rows, or orchestration metadata.

## Run-plan format

A run plan is a UTF-8 YAML file. The top-level value must be a mapping, unknown
fields are rejected, and `version` must be `1`.

### Top-level fields

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `version` | integer | Yes | - | Must be `1` |
| `resources` | [resource mapping](#resources) | No | `threads: auto`, `device: auto` | Host-resource policy |
| `datasets` | mapping of names to [object specifications](#object-specifications) | No | `{}` | Dataset objects |
| `models` | mapping of names to object specifications | No | `{}` | Model objects |
| `metrics` | mapping of names to object specifications | No | `{}` | Metric objects |
| `tasks` | list of [tasks](#tasks) | Yes | - | At least one capability invocation |

<!-- markdownlint-enable MD013 -->

### Resources

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Field | Type | Default | Behavior |
| --- | --- | --- | --- |
| `threads` | `auto` or positive integer | `auto` | `auto` uses every CPU allowed by CPU affinity and cgroup quota; an integer is capped at that count |
| `device` | `auto`, `cpu`, `cuda`, or `cuda:N` | `auto` | `auto` selects `cuda:0` when CUDA is visible, otherwise `cpu` |

<!-- markdownlint-enable MD013 -->

`cuda` means `cuda:0`. Requesting a CUDA device that is not visible fails the
run with exit status `2`. At most one device is used for the whole run.

The thread budget sets PyTorch's thread count and limits the BLAS and OpenMP
thread pools that NumPy, SciPy, and PyTorch load. It does not start processes
or run tasks concurrently. ONNX models ignore it: CheckMAITE's ONNX models
create their ONNX Runtime session with default options, so ONNX Runtime
chooses its own thread count.

### Object specifications

Each entry under `datasets`, `models`, or `metrics` has this form:

```yaml
name:
  class: package.module.ClassName
  args: {}
```

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `class` | non-empty string | Yes | - | Import path or [plugin reference](#plugins) for a trusted class or factory function |
| `args` | mapping | No | `{}` | Keyword constructor arguments |

<!-- markdownlint-enable MD013 -->

An argument that is itself an object uses `_class` instead of `class`, so that
ordinary arguments can contain a `class` key:

```yaml
metrics:
  map50:
    class: checkmaite.core.object_detection.metrics.TorchODMetric
    args:
      od_metric:
        _class: torchmetrics.detection.MeanAveragePrecision
        args:
          box_format: xyxy
          iou_type: bbox
      return_key: map_50
      metric_id: map50
```

A mapping is treated as a nested object only when its keys are `_class` and,
optionally, `args`.

The resolved device is passed as a `device` argument to model constructors that
accept one (or accept `**kwargs`), unless the plan already sets it. Dataset,
metric, and nested-object constructors never receive it.

Dataset and model file formats are not fixed by the runtime. They are defined
by the selected CheckMAITE class or plugin. For example, the sample plan reads
COCO data and an ONNX model because it selects the COCO dataset loader and the
ONNX model class.

### Tasks

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Field | Type | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `name` | string | Yes | - | Unique output name of letters, digits, `_`, `.`, and `-`, starting with a letter or digit |
| `capability` | non-empty string | Yes | - | Built-in, import-path, or plugin capability reference |
| `capability_args` | mapping | No | `{}` | Capability constructor arguments |
| `dataset` | name, list of names, or null | No | `null` | Objects selected from `datasets` |
| `model` | name, list of names, or null | No | `null` | Objects selected from `models` |
| `metrics` | name, list of names, or null | No | `null` | Objects selected from `metrics` |
| `config` | mapping | No | `{}` | Capability-specific run configuration |
| `use_cache` | boolean | No | `true` | Reuse cached predictions and evaluations from earlier runs |
| `report_threshold` | number | No | `0.5` | Threshold passed to Markdown report generation |

<!-- markdownlint-enable MD013 -->

Every name a task references must exist in the corresponding top-level mapping.
The capability decides which object kinds, and how many of each, a task
accepts.

`builtin:MaiteEvaluation` selects a built-in capability and infers the problem
type from the task's CheckMAITE objects. The explicit forms
`builtin:object_detection.Name` and `builtin:image_classification.Name` are
needed when every object in the task comes from a plugin.

### Plugins

A plugin is a trusted Python file referenced as `file:<path>:<Name>`, for
example `file:plugins/detector.py:FixedBoxDetector`. It can be used anywhere a
`class`, `_class`, or `capability` reference is accepted.

- **Paths** are resolved from the directory containing the plan unless they are
  absolute. The text after the last `:` names the class or factory function.
- **One self-contained file.** The file is imported directly from its path. It
  is not added to `sys.path`, so it cannot import a sibling file or use relative
  imports.
- **Dependencies must already be installed.** A plugin can import CheckMAITE
  and any installed package; nothing is installed at run time.
- **Loaded once per run.** Every reference to the same file shares one module.

Plugins and import paths run with the full permissions of the process. They are
not sandboxed.

## Execution

The command validates the plan, then builds every dataset, model, and metric
that some task references. Objects no task references are logged and skipped.
Tasks then run one after another, in the order listed, in a single process.
There is no retry, resume, or checkpointing.

The first failure stops the run. If task 3 fails, tasks 4 onward don't run.
Earlier tasks' outputs stay on disk.

## Outputs and exit status

A successful plan writes:

```text
OUTPUT_DIR/
├── run-results.json
├── analytics/
└── tasks/
    └── TASK_NAME/
        ├── run.json
        └── REPORT_FILE
```

`run-results.json` records each task's run UID, capability ID, run file, and
report file or URI. It is removed when a run starts and written only when every
task succeeds, so a failed run has no summary. Analytics are written as Parquet
files at the end of a successful run. Analytics files from earlier runs in the
same output directory are not removed. A capability that has no Markdown report
still writes `run.json`.

<!-- markdownlint-disable MD013 -- table rows cannot be wrapped -->

| Exit status | Meaning |
| --- | --- |
| `0` | Every task succeeded |
| `1` | A constructor or task failed while running, including on bad input data, or the host could not be inspected |
| `2` | The plan, arguments, imports, or configuration are invalid |

<!-- markdownlint-enable MD013 -->

Constructor arguments are checked against the constructor's signature before it
is called, so an unknown or missing argument exits `2`. Every task's capability,
config, object names, and number of datasets, models, and metrics are checked
before any object is built or any task runs, so those errors also exit `2`. Logs
go to standard error and include a timestamp, severity, logger name, and
message. Python warnings are logged as `WARNING` records from the `py.warnings`
logger.

## Hardware, storage, and network

Actual requirements depend on the selected model, dataset, metrics, and batch
size. A minimum practical CPU invocation needs one CPU, enough memory for the
selected objects, and storage for the approximately 2.64 GiB unpacked CPU
container plus input, output, and cache data. As a starting point for non-trivial
evaluation, allocate
4 CPUs, 16 GiB of memory, and storage sized for the container plus at least
twice the working dataset/model footprint. Measure the real workload before
setting production limits.

CUDA execution needs an AMD64 host, a compatible NVIDIA GPU and driver, and GPU
memory sufficient for the selected model and batch. The CUDA container is
approximately 9.55 GiB unpacked before input, output, and cache data. No physical
CUDA
performance or minimum-memory claim can be inferred from package metadata;
validate the intended workload on the target GPU.

The built-in runtime needs no network when all artifacts and plugin dependencies
are mounted or already included. A selected class or trusted plugin may require
network access for a remote object store or model service. Prefer mounting or
pre-populating large artifacts. If network access is necessary, allow only the
required destinations.

## Hardened deployment

A Docker invocation with the expected least-privilege controls is:

```bash
docker run --rm \
  --read-only \
  --user 10001:10001 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --network none \
  --mount type=bind,source="$PWD/input",target=/checkmaite,readonly \
  --mount type=bind,source="$PWD/output",target=/output \
  --mount type=bind,source="$PWD/cache",target=/cache \
  checkmaite-batch:cpu
```

Remove `--network none` only when a selected class or plugin has a documented
network dependency. On Kubernetes, set `runAsNonRoot: true`, `runAsUser: 10001`,
`runAsGroup: 10001`, `readOnlyRootFilesystem: true`,
`allowPrivilegeEscalation: false`, and drop `ALL` capabilities. Mount only
`/output` and `/cache` writable, mount inputs and secrets read-only, and apply a
default-deny NetworkPolicy with narrow egress exceptions when required.

## Publication and verification

Release pipelines build and exercise the supported product platforms with
Docker 25 or later, retain unfiltered SPDX SBOM and vulnerability reports, and
block publication on every Medium, High, or Critical finding. Unfixed findings
are included; publication requires remediation or a formal program exception.
Exceptions are recorded in `docker/ci/dsor3-exceptions.trivyignore.yaml`, each
with its approved exception ID and an expiry date, and apply only to the
release gate. The release scan refuses to run while any entry is still marked
`PENDING`. Because releases rebuild from the tag, commit exceptions before
creating it.

Before enabling releases, configure the program Harbor project to make Semantic
Version tags immutable, approve the digest-pinned Docker Official Ubuntu,
NVIDIA, and Astral base-container sources (or record the required CS-1-S-2
exception), and provide the Harbor credentials and AWS KMS key to protected CI
jobs.

A release is started from the default branch with `RELEASE_TAG` set to an
existing unprefixed final Semantic Version Git tag, such as `1.2.3`; pre-release
suffixes are rejected. The tag's commit must be on `main`. The release job checks
out that commit and runs the tag's own Dockerfile, `docker/ci/` scripts, and
exception file, not the copies on `main`. A tag is therefore publishable only if
it already contains `docker/ci/`; tags created before container publishing was
added cannot be published. A fix to the release tooling requires a new version
tag. Set
`CONTAINER_RELEASE_DRY_RUN=true` for the required first staging exercise; CI
then pushes, signs, attests, verifies, and cleans up without creating a Semantic
Version tag. Publication creates these write-once references:

```text
harbor.jatic.net/openteams/checkmaite/cpu:VERSION
harbor.jatic.net/openteams/checkmaite/cuda:VERSION
```

Both references are Linux AMD64 images. CI refuses to overwrite an existing
version tag. Each image digest is signed with Cosign through the program AWS KMS
key, and its SPDX SBOM and Trivy vulnerability report are attached as signed
attestations. CI uses an explicit Cosign configuration with no public
transparency-log or timestamp service.

The signature and attestations are verified before the write-once Semantic
Version tag is applied as the final publication action. Before the first
release, exercise the complete flow against a non-release Harbor staging tag
with the protected KMS key. Deployments should pin the reported image digest
rather than relying on a mutable convenience tag.

## Licences

CheckMAITE and the container runtime are licensed under Apache-2.0. Each
container also redistributes third-party software under its own terms:

- Ubuntu packages, under the licences recorded in each package's
  `/usr/share/doc/<package>/copyright` file, which the image retains.
- Python packages in `/opt/venv`, under the licences in their `.dist-info`
  metadata, including PyTorch (BSD-3-Clause) and Ray (Apache-2.0).
- In the CUDA container only, NVIDIA CUDA, cuDNN, and NCCL, under the
  [NVIDIA Deep Learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf)
  (included in the image as `/NGC-DL-CONTAINER-LICENSE`) and the CUDA EULA.
  These are proprietary. The CUDA image's `org.opencontainers.image.licenses`
  label is therefore `Apache-2.0 AND LicenseRef-NVIDIA-Proprietary`.

The SBOM published with each image lists every package and its declared
licence.
