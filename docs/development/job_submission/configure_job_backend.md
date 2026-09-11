# `configure_job_backend(...)`

This page documents the `configure_job_backend(...)` contract used by job submission.

`configure_job_backend(...)` selects the active job backend and captures the deployment-specific settings that cannot be inferred from an individual `submit_capability(...)` call.

## Why backend configuration is required

Synchronous `capability.run(...)` executes in the caller's Python process. That process already knows its local filesystem, cache, credentials, installed packages, and result storage location.

Job submission separates those concerns:

- the **client** submits work and later observes job state,
- the **job backend** decides where and how that work runs,
- **workers** execute capability code in a potentially different process, node, container, or cluster,
- and durable outputs must be written somewhere both workers and clients can access.

So the backend needs explicit configuration for the execution environment and coordination boundary. Depending on the backend, that includes:

- which backend implementation to use (`"ray"`, `"ray-simple"`, or future backends),
- where to connect for execution (`address` for Ray clusters),
- how workers should be prepared (`runtime_env`),
- where workers should persist analytics results (`analytics_store`),
- that job-submission workers run with capability-local caching disabled
  (`use_cache=False`) because worker-local caches are ephemeral and not shared,
- which clients should share job identity, dedupe, and reattach behavior (`idempotency_scope` for the registry-backed Ray backend),
- which native Ray namespace owns the shared backend state
  (`registry_namespace`; each namespace has one fixed internal registry actor),
- actor pending-call limits (`registry_max_pending_calls`, `controller_max_pending_calls`) for handles returned when this client creates an actor; Ray cannot apply them to reattached handles,
- and operational settings such as timeouts, retention, cleanup, and actor resource placement.

Provenance fields such as `user_id`, `workspace_id`, `environment`, `executor`,
`cluster_id`, and `request_id` are not backend settings. Set them with
`checkmaite.configure_provenance(...)`. Job backends add job-specific fields such
as `job_id`, backend name, and submission/completion times when they write run
history.

The goal is to make the backend's execution, coordination, storage, and operational assumptions explicit before jobs are submitted.

## API shape

```python
from checkmaite.jobs import configure_job_backend

configure_job_backend(
    "ray",
    address="ray://cluster-head:10001",
    idempotency_scope="my-workspace-or-experiment",
    runtime_env={
        "working_dir": ".",
        "env_vars": {"MODEL_REGISTRY_URL": "https://registry.internal"},
    },
    analytics_store={
        "backend": "parquet",
        "uri": "s3://team-checkmaite/analytics-store",
        # optional:
        # "storage_options": {...},
    },
)
```

The first positional argument selects the job backend. All other keyword arguments are backend-specific configuration forwarded to that backend's constructor.

Ray job backend choices:

- `"ray"`: registry/controller-backed Ray job backend for shared clusters, dedupe,
  and reattach across client restarts. Requires `idempotency_scope`.
- `"ray-simple"`: direct Ray task-based job backend for local or single-driver workflows.
  It does not use a registry and does not take `idempotency_scope`.

## Examples

### Local development with reattachable jobs

```python
configure_job_backend(
    "ray",
    address="local",
    idempotency_scope="local-dev",
    analytics_store={"backend": "parquet", "uri": "./analytics_store"},
)
```

### Local development with the simple Ray job backend

```python
configure_job_backend(
    "ray-simple",
    address="local",
    analytics_store={"backend": "parquet", "uri": "./analytics_store"},
)
```

### Shared Ray cluster with object storage

```python
configure_job_backend(
    "ray",
    address="ray://cluster-head:10001",
    idempotency_scope="team-a-prod-evals",
    runtime_env={
        "working_dir": ".",
        "env_vars": {"AWS_REGION": "us-east-1"},
    },
    analytics_store={
        "backend": "parquet",
        "uri": "s3://team-checkmaite/analytics-store",
        "storage_options": {"anon": False},
    },
)
```

### Shared status + reattach configuration

```python
configure_job_backend(
    "ray",
    address="ray://cluster-head:10001",
    analytics_store={"backend": "parquet", "uri": "s3://team-checkmaite/analytics-store"},
    idempotency_scope="team-a-notebooks",
    # Use a different Ray namespace when an independent registry is required.
    registry_namespace="checkmaite_jobs",
    controller_retention_s=3600,
    max_retained_terminal_controllers=1000,
)
```

`registry_namespace` uses Ray's built-in named-resource namespace. It does not
create a new Ray cluster, Kubernetes namespace, resource quota, or security
boundary. Checkmaite's registry actor name is fixed and internal; users select
an independent registry by selecting another Ray namespace. All clients sharing
a namespace must use compatible registry settings.

For detailed Ray runtime behavior, see [Ray job backend](ray_job_backend.md) and [Ray simple job backend](ray_simple_job_backend.md). For worker image and cluster environment guidance, see [Worker environments](worker_environments.md). For store semantics, provenance, and URI resolution details, see [Distributed analytics store](analytics_store.md).
