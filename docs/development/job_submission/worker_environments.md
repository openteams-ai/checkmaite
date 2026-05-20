# Ray worker environments

This page is aimed primarily at the **platform team** operating Ray clusters for `checkmaite` job submission.

The short version is:

- `checkmaite` does **not** build worker environments for you,
- the platform owns the worker image and cluster spec,
- and the job backend can only supply a Ray `runtime_env` overlay at connection time.

Both `"ray"` and `"ray-simple"` use Ray workers and the same `runtime_env` mechanics. The default `"ray"` job backend also runs registry/controller actors, while `"ray-simple"` submits direct Ray tasks from one driver.

## Two practical modes

### Local development mode

For local development, Ray workers run from the developer's current Python environment. Use `"ray"` when you want registry-backed reattach behavior, or `"ray-simple"` when a process-local direct Ray task-based job backend is enough.

```python
configure_job_backend(
    "ray",
    address="local",
    idempotency_scope="local-dev",
    analytics_store={"backend": "parquet", "uri": "./analytics_store"},
)
```

This is the lightest-weight setup and is what the walkthrough notebook demonstrates.

### Platform / cluster mode

For shared infrastructure, workers should start from a platform-managed base image.

That image should already contain the heavy, stable parts of the environment:

- Python
- Ray
- `checkmaite`
- CUDA / PyTorch when GPUs are involved
- system libraries needed by capabilities and model code
- storage connectors needed by your deployment (for example `s3fs`, `gcsfs`, `adlfs`)

## Recommended Docker model

The current code is best thought of as a **platform image + Ray overlay** model.

### Base image responsibilities

The base image should provide everything needed for workers to import and execute capability code reliably.

A practical layering strategy is:

1. base OS and security hardening,
2. Python and Ray,
3. CUDA / PyTorch stack if needed,
4. pinned `checkmaite` version and its heavy dependencies,
5. storage and platform integration libraries.

That keeps the worker startup path predictable and avoids reinstalling the expensive parts of the environment on every task.

### Ray `runtime_env` responsibilities

Use `runtime_env` for smaller, faster-changing overlays such as:

- environment variables,
- a `working_dir` or `py_modules` bundle for iterative code updates,
- small supplemental Python packages.

Example:

```python
configure_job_backend(
    "ray",
    address="ray://cluster-head:10001",
    idempotency_scope="team-a-prod-evals",
    runtime_env={
        "working_dir": ".",
        "pip": ["my-small-lib==0.3.1"],
        "env_vars": {
            "MODEL_REGISTRY_URL": "https://registry.internal",
        },
    },
    analytics_store={
        "backend": "parquet",
        "uri": "s3://team-checkmaite/analytics-store",
        "storage_options": {"anon": False},
    },
)
```

## How this maps to the current code

Both Ray job backends accept:

- Ray connection and environment options through `configure_job_backend(..., **kwargs)`
- analytics-store configuration through the explicit `analytics_store=...` argument

Those concerns are separate on purpose:

- `runtime_env` controls how Ray workers are prepared,
- `analytics_store` tells workers where durable run data should be written.

## Platform-team checklist

For a production cluster, make sure workers can:

1. **Import the same code** the client expects
   - capability classes must be importable on workers,
   - run models must deserialize correctly,
   - version skew between client and workers should be avoided.

2. **Access input data and models**
   - dataset URIs must resolve from worker nodes,
   - model artifacts must be reachable from worker nodes,
   - credentials must be present in the worker environment.

3. **Access the analytics store**
   - workers must be able to write to the configured store URI,
   - the client must also be able to read from that same durable location later.

4. **Expose the right compute resources**
   - CPU and GPU resources must be visible to Ray,
   - and the cluster should be sized for the expected capability mix.

The default `"ray"` job backend also needs the worker image to import the registry/controller code. The `"ray-simple"` job backend only needs the worker task code and submitted capability dependencies.

## Example: object-store analytics store

The current jobs analytics-store configuration supports the Parquet backend with a URI and optional storage options.

```python
configure_job_backend(
    "ray",
    address="ray://cluster-head:10001",
    idempotency_scope="team-a-prod-evals",
    analytics_store={
        "backend": "parquet",
        "uri": "s3://team-checkmaite/results",
        "storage_options": {
            "anon": False,
        },
    },
    runtime_env={
        "env_vars": {
            "AWS_REGION": "us-east-1",
        }
    },
)
```

For this to work in practice:

- workers need credentials that can write to that bucket,
- the client needs credentials that can later read the same bucket,
- and the worker image must include the storage dependencies required by the deployment.

## Practical guidance

### Prefer heavy dependencies in the image

Put large and slow-moving dependencies in the image:

- `checkmaite`
- PyTorch / CUDA
- large model-serving dependencies
- storage connectors used everywhere

### Use `runtime_env` for deltas, not full environments

Ray can install packages via `runtime_env["pip"]`, but using that for entire heavyweight environments increases cold-start time and operational variability.

### Pin versions across client and worker

The client serializes capability objects and expects workers to import compatible code. Loose versioning can create subtle failures. Treat the worker image, the client environment, and any `runtime_env` overlay as one versioned deployment unit.

## Current limitations

The current job backend deliberately stops short of becoming a packaging system.

It does **not**:

- build Docker images,
- publish environments,
- manage lockfiles for the platform,
- or guarantee cross-cluster compatibility automatically.

That work belongs in platform tooling, cluster configuration, and release discipline.
