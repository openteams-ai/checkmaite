# Job submission and cluster execution

`CheckMAITE` traditionally executes capabilities through `capability.run(...)`,
which blocks until the run finishes. For small local workloads that is fine. For
long-running evaluations, it creates two problems:

1. **Interactivity** — notebook users cannot keep working smoothly while a run
   is executing.
2. **Compute scaling** — one local Python process is a poor fit for capabilities
   that need more CPU/GPU resources or cluster execution.

The job-submission subsystem addresses both problems:

- it gives users a **non-blocking job handle**,
- it lets the same API target **local or distributed job-submission backends**.

## How it works

<div class="grid cards" markdown>

- [**Protocol and lifecycle** :octicons-arrow-right-24:](protocol.md)

  The shared job handle contract, lifecycle states, reference-first results, and
  error semantics.

- [**Kubernetes and KubeRay** :octicons-arrow-right-24:](kubernetes.md)

  Kubernetes-specific guidance for KubeRay placement, detached actors,
  autoscaling, and durability boundaries.

- [**Distributed analytics store** :octicons-arrow-right-24:](analytics_store.md)

  Why durable result writes are more subtle in distributed execution and what
  job submission expects from the configured store.

</div>

For backend settings, see [Job backend configuration](configure_job_backend.md).
For how each backend runs jobs and how workers are set up, see [Ray job
backend](ray_job_backend.md), [Ray simple job
backend](ray_simple_job_backend.md), and [Worker
environments](worker_environments.md). To submit a job step by step, follow the
[Ray Simple Job
Submission](../../tool-usage/ray_simple_job_submission_tutorial.ipynb)
and [Ray Job
Submission](../../tool-usage/ray_job_submission_tutorial.ipynb) tutorials.
