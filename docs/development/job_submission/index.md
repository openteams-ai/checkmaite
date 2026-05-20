# Job submission and cluster execution

`checkmaite` traditionally executes capabilities through `capability.run(...)`, which blocks until the run finishes. For small local workloads that is fine. For long-running evaluations, it creates two problems:

1. **Interactivity** — notebook users cannot keep working smoothly while a run is executing.
2. **Compute scaling** — one local Python process is a poor fit for capabilities that need more CPU/GPU resources or cluster execution.

The job-submission subsystem addresses both problems:

- it gives users a **non-blocking job handle**,
- it lets the same API target **local or distributed job-submission backends**.

## What to read next

<div class="grid cards" markdown>

- [__Protocol and lifecycle__ :octicons-arrow-right-24:](protocol.md)

  The shared job handle contract, lifecycle states, reference-first results, and error semantics.

- [__Job backend configuration (`configure_job_backend`)__ :octicons-arrow-right-24:](configure_job_backend.md)

  What backend-level settings must be configured before submission, including execution target, worker environment, storage, and shared job identity.

- [__Ray job backend__ :octicons-arrow-right-24:](ray_job_backend.md)

  The default registry/controller-backed Ray job backend for reattachable jobs.

- [__Ray simple job backend__ :octicons-arrow-right-24:](ray_simple_job_backend.md)

  The direct process-local Ray task-based job backend for simple single-driver workflows.

- [__Worker environments__ :octicons-arrow-right-24:](worker_environments.md)

  Guidance for platform teams on container images, Ray worker setup, and `runtime_env` overlays.

- [__Kubernetes and KubeRay__ :octicons-arrow-right-24:](kubernetes.md)

  Kubernetes-specific guidance for KubeRay placement, detached actors, autoscaling, and durability boundaries.

- [__Distributed analytics store__ :octicons-arrow-right-24:](analytics_store.md)

  Why durable result writes are more subtle in distributed execution and what job submission expects from the configured store.

</div>
