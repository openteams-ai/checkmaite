# CheckMAITE container runtime

This is a private helper package for the batch container. It is built and
installed by the root `Dockerfile`; it is not included in the published
`checkmaite` wheel or source distribution.

User documentation for the command line and the run-plan format is published
as the [batch container reference](../../docs/reference/container.md). This
README covers design notes and maintenance.

## Design

The batch container runs a **run plan**: one YAML file that declares everything
the container will execute. It lists the datasets, models, metrics, resources,
and the tasks (capabilities) to run with them. The container reads the plan,
builds every object a task references, runs the tasks in order, writes the
results to the output mount, and exits. Objects no task references are logged
and skipped.

The plan format is defined by Pydantic models in
`src/checkmaite_container/_plan.py`:

- **Validation:** every plan is validated against these models before anything
  runs. An invalid plan exits with status `2`.
- **Schema:** `generate_schema.py` generates `schema/run-plan-v1.schema.json`
  from the same models. Editors and YAML tooling can use it for completion and
  validation. See [Run-plan schema](#run-plan-schema).

`docker/example-run.yaml` is an example plan; tests run it unchanged on small
repository fixtures.

## Assumptions

### Execution model

- **One host, one process.** Nothing is distributed and no worker processes are
  started.
- **Serial tasks.** Tasks run one after another, in the order listed. There's no
  parallelism between tasks; the thread setting only affects parallelism inside
  PyTorch and the numerical libraries.
- **Finite batch job.** It runs the plan once and exits. It isn't a service, so
  there's no retry, resume or checkpointing.
- **The first failure stops the run.** If task 3 fails, tasks 4 onward don't
  run. Earlier tasks' `run.json` and reports stay on disk. `run-results.json`
  is removed when a run starts and written only when every task succeeds, so a
  failed run has no summary. Analytics files from earlier runs in the same
  output directory are not removed.
- **Exit codes.** `0` means every task succeeded. `2` means the plan,
  arguments, imports, or configuration are invalid. Constructor arguments are
  checked against the constructor's signature before it is called. `1` means
  a constructor or task failed while running, including on bad input data.

### Resources

- **All available CPUs by default.** `threads: auto` means every CPU the
  container can actually use, after CPU affinity and cgroup limits.
- **One thread budget for the process.** `threads` sets the PyTorch thread
  count and limits the BLAS and OpenMP pools that NumPy, SciPy, and PyTorch
  load. ONNX models ignore it: CheckMAITE's ONNX models create their ONNX
  Runtime session with default options, so ONNX Runtime picks its own thread
  count.
- **One device.** `auto` picks `cuda:0` if a GPU is visible, otherwise `cpu`.
  Multi-GPU isn't supported; at most one device is used for the whole run.
- **Device goes to models only.** It's passed to a model constructor only if the
  constructor accepts a `device` argument (or `**kwargs`) and the plan didn't
  set one. Datasets and metrics never receive it.
- **`--batch-size` applies to every task** whose capability config has a
  `batch_size` field, not selectively.

### Trust and environment

- **All referenced code is trusted.** Dotted import paths and `file:` plugins run
  with the process's full permissions, with no sandbox.

### Caching

- **Caching is on by default** (`use_cache: true`), so results in the cache
  directory are reused by later runs.

## Object references

Each dataset, model, and metric names its constructor with `class` and passes
keyword arguments with `args`. An argument that is itself an object uses
`_class` instead, so ordinary arguments can still contain a `class` key:

```yaml
metrics:
  map50:
    class: checkmaite.core.object_detection.metrics.TorchODMetric
    args:
      od_metric:
        _class: torchmetrics.detection.MeanAveragePrecision
        args:
          box_format: xyxy
      return_key: map_50
```

A mapping is treated as a nested object only when its keys are `_class` and,
optionally, `args`. Nested objects never receive the device.

## Plugins

A plugin is a trusted Python file referenced as `file:<path>:<Name>`, for
example `file:plugins/detector.py:FixedBoxDetector`. It can be used anywhere a
`class`, `_class`, or `capability` reference is accepted.

- **One self-contained file.** The file is imported directly from its path. It
  is not added to `sys.path` and is not part of a package, so it cannot import a
  sibling file (`import helper`) or use relative imports (`from . import x`).
  Put everything the plugin needs in that one file.
- **Dependencies must already be installed.** A plugin can import CheckMAITE
  and any package in the container, but nothing is installed at run time.
- **Paths** are resolved from the directory containing the plan unless they are
  absolute. The text after the last `:` names the class or factory function.
- **Loaded once per run.** Every reference to the same file shares one module.
- **Explicit `builtin:` forms for plugin-only tasks.** Short `builtin:` names
  infer the problem type from CheckMAITE objects only, so a task whose objects
  all come from plugins needs, for example,
  `builtin:object_detection.MaiteEvaluation`.

## Dependencies and lock

The containers need the CPU (`+cpu`) or CUDA (`+cu130`) PyTorch build, which
only PyTorch's own indexes provide. uv can select them with extras plus
`[tool.uv.sources]`, but that section is uv-only and is dropped from published
package metadata. `cpu` and `cuda` extras in CheckMAITE's root
`pyproject.toml` would therefore install the wrong PyTorch build for anyone
using them with pip:

```bash
# Linux x86_64, with root extras cpu = ["torch==2.13.0"] + a tool.uv.sources CPU index
pip install "checkmaite[cpu]"   # or: uv pip install "checkmaite[cpu]"
# -> installs PyPI torch 2.13.0, the CUDA 13 build, plus nvidia-*-cu13 packages
```

So the `cpu` and `cuda` extras, their index rules, and the
`checkmaite-container` entry point live in this unpublished package, which has
its own lock. With the `cpu` extra the lock resolves `+cpu` wheels without
CUDA, NVIDIA, or Triton packages; with `cuda` it resolves `+cu130` wheels and
`onnxruntime-gpu`.

PyYAML and threadpoolctl are direct dependencies only here. CheckMAITE already
carries both transitively, at the same versions.
`docker/ci/check_runtime_lock.py` requires every package in this lock to be in
the root lock at the same version.

To refresh the private lock while retaining versions already selected by the
main project lock (use the uv version pinned in `.gitlab-ci.yml`; older uv
releases rewrite the whole lock):

```bash
cp uv.lock docker/runtime/uv.lock
uv lock --directory docker/runtime --python 3.12
uv run python docker/ci/check_runtime_lock.py
```

See the [container maintenance runbook](../../docs/development/container_maintenance.md)
for the complete base, snapshot, lock, tool, scan, and validation workflow.

## Run-plan schema

The Pydantic models in `src/checkmaite_container/_plan.py` define the run-plan
fields. Generate the versioned JSON Schema used by YAML tooling with:

```bash
python docker/runtime/generate_schema.py
```

The schema is committed at `schema/run-plan-v1.schema.json`. Tests run the
generator with `--check` so model changes cannot leave the schema stale.
