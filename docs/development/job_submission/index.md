# Job submission and cluster execution

`checkmaite` traditionally executes capabilities through `capability.run(...)`, which blocks until the run finishes. For small local workloads that is fine. For long-running evaluations, it creates two problems:

1. **Interactivity** — notebook users cannot keep working smoothly while a run is executing.
2. **Compute scaling** — one local Python process is a poor fit for capabilities that need more CPU/GPU resources or cluster execution.

The job-submission subsystem addresses both problems:

- it gives users a **non-blocking job handle**,
- it lets the same API target **local or distributed Ray execution**,
- and it returns a **lightweight reference** to durable results rather than bouncing the full run object back through the client.

## What to read next

<div class="grid cards" markdown>

- [__Backend configuration (`configure_backend`)__ :octicons-arrow-right-24:](configure_backend.md)

  Why explicit `analytics_store` config is required in distributed execution and how that config is forwarded from client to worker.

- [__Protocol and lifecycle__ :octicons-arrow-right-24:](protocol.md)

  Why the jobs protocol exists, why `result()` is reference-first, and how lifecycle semantics work.

- [__Ray backend__ :octicons-arrow-right-24:](ray_backend.md)

  Why Ray was chosen, how the backend maps onto Ray Core, and how to use it end-to-end.

- [__Worker environments__ :octicons-arrow-right-24:](worker_environments.md)

  Guidance for platform teams on container images, Ray worker setup, and `runtime_env` overlays.

- [__Distributed analytics store__ :octicons-arrow-right-24:](analytics_store.md)

  Why durable writes are more subtle in distributed execution and how the current implementation handles idempotency and URI resolution.

- [__Walkthrough notebook__ :octicons-arrow-right-24:](../job_submission_walkthrough.ipynb)

  Executable maintainer walkthrough showing the current backend behavior in practice.

</div>

## Current implementation scope

Today the jobs subsystem is intentionally narrow:

- backend: **Ray Core**
- result type: **`CapabilityRunRef`**
- durable store configuration: **explicitly required** via `configure_backend(..., analytics_store=...)`
- artifact hydration: **not implemented yet** (`outputs_uri` is expected to be `None`)

These pages document the current code, not a future architecture sketch.
