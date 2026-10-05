# Explanations

Background on how CheckMAITE works and why it is built that way. These pages don't walk through a task; for that, see the [tutorials](../tool-usage/index.md) and [how-to guides](../how-to/index.md).

<div class="grid cards" markdown>

-  [__Key concepts: capabilities, runs, and caches__ :octicons-arrow-right-24:](../development/key_concepts.md)

    The core abstractions behind how evaluations are defined, executed, and cached.

-  [__Job submission overview__ :octicons-arrow-right-24:](../development/job_submission/index.md)

    Why CheckMAITE has non-blocking job handles and pluggable execution backends.

-  [__Job protocol and lifecycle__ :octicons-arrow-right-24:](../development/job_submission/protocol.md)

    The shared job handle contract, lifecycle states, reference-first results, and error semantics.

-  [__Ray job backend__ :octicons-arrow-right-24:](../development/job_submission/ray_job_backend.md)

    Why Ray, the registry-backed execution model, status mapping, and design trade-offs.

-  [__Ray simple job backend__ :octicons-arrow-right-24:](../development/job_submission/ray_simple_job_backend.md)

    How the process-local Ray backend runs and tracks tasks from a single driver.

-  [__Worker environments__ :octicons-arrow-right-24:](../development/job_submission/worker_environments.md)

    How container images, worker setup, and `runtime_env` overlays fit together.

-  [__Kubernetes and KubeRay__ :octicons-arrow-right-24:](../development/job_submission/kubernetes.md)

    KubeRay placement, detached actors, autoscaling, and durability boundaries.

-  [__Analytics store in distributed execution__ :octicons-arrow-right-24:](../development/job_submission/analytics_store.md)

    Why durable result writes are more subtle in distributed execution, and what job submission expects from the store.

</div>
