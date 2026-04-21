# `configure_backend(...)` and analytics-store handoff

This page documents the `configure_backend(...)` contract used by job submission.

## Why this configuration is required

In local synchronous execution, the process that computes the run already knows:

- where the analytics store lives,
- how to write to it,
- and how to read it back later.

In distributed job submission, those responsibilities are split:

- the **client** chooses the durable store location,
- the **worker** needs that information so it can persist results,
- and the **client** later needs a stable way to find/read the payload data the worker wrote.

So `configure_backend(...)` requires explicit `analytics_store=...` configuration.

## API shape

```python
from checkmaite.jobs import configure_backend

configure_backend(
    "ray",
    analytics_store={
        "backend": "parquet",
        "uri": "./analytics_store",
        # optional:
        # "storage_options": {...},
    },
)
```

Current config model (`AnalyticsStoreConfig`):

- `backend`: currently only `"parquet"`
- `uri`: required store root URI/path
- `storage_options`: optional mapping passed through to backend I/O

## Producer/consumer handoff in current code

This is the communication path that bridges consumer (client) and producer (worker):

1. `checkmaite.jobs._api.configure_backend(...)` accepts `analytics_store`.
2. `RayBackend` validates/stores that config once.
3. `checkmaite.jobs._api.submit_capability(...)` delegates to the active backend for each job submission.
4. `RayBackend.submit_capability(...)` injects the stored config into task kwargs (`_analytics_store`).
5. Worker task (`_execute_capability_ref`) pops `_analytics_store`, builds a worker-side `AnalyticsStore`, and writes records.

That is how workers write to a client-selected durable location that the client can later query/read.

## Examples

### Local development

```python
configure_backend(
    "ray",
    address="local",
    analytics_store={"backend": "parquet", "uri": "./analytics_store"},
)
```

### Shared object storage

```python
configure_backend(
    "ray",
    address="ray://cluster-head:10001",
    analytics_store={
        "backend": "parquet",
        "uri": "s3://team-checkmaite/analytics-store",
        "storage_options": {"anon": False},
    },
)
```

For broader Ray runtime/reconfiguration behavior (`runtime_env`, `force_reinit`), see [Ray backend](ray_backend.md). For store semantics and URI resolution details, see [Distributed analytics store](analytics_store.md).
