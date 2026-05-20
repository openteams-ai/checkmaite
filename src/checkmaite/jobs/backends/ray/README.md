# Ray job backend

`ray` is the production Ray job backend for `checkmaite` jobs:

```python
from checkmaite.jobs import configure_job_backend

configure_job_backend(
    "ray",
    analytics_store={"backend": "parquet", "uri": "./job-results"},
    idempotency_scope="my-workspace-or-experiment",
)
```

The `ray` job backend stores (small) job metadata in a `JobRegistryActor` and
starts one `JobControllerActor` for each new capability run. In Ray, an actor is
a stateful Python object that runs in the cluster as its own worker process. A
detached actor is named and can keep running after the notebook or driver that
created it exits.

The controller actor owns the live Ray worker task: it keeps the task reference
in the cluster, monitors completion, handles cancellation, and writes final
status and result metadata back to the registry. Because the task is owned by a
cluster-side actor instead of the notebook process, the job can keep running
after the submitting client exits as long as the Ray cluster and actors stay
alive. The registry makes job ids, duplicate submissions, and finished job
status visible to other clients that use the same scope.

Use `ray` when you need jobs to keep running after a notebook or driver exits,
when multiple clients need to list or reconnect to jobs, or when duplicate
submissions should converge to an existing logical run. The `ray` job backend is
intentionally more operationally involved than `ray-simple`.

> [!IMPORTANT]
> The `ray` job backend can reconnect to jobs after your notebook or driver
> exits, as long as the Ray cluster and the job backend's Ray processes are still
> running. It does not permanently save job tracking information by itself.
> If the Ray cluster is shut down, the registry process is lost, or incompatible
> code is deployed, job metadata will disappear because the registry state is
> lost, even if a worker already wrote data to the analytics store.

## Practical guidance

Use `ray` when jobs should keep running after a notebook or driver exits, when
more than one client needs to find or reconnect to the same jobs, or when repeat
submissions should reuse the same existing job instead of starting duplicate
work. It is a good fit for shared Ray clusters, long-running notebooks, KubeRay,
and capability jobs that run for minutes or hours.

Use `ray-simple` when one Python process or notebook owns the whole workflow,
losing job handles after that process exits is acceptable, and duplicate
submissions are either safe or handled by your own code.

To use `ray` well:

- choose one stable job namespace for each workspace or experiment; this is the
  `idempotency_scope` value in the job backend config;
- make clients use the same job namespace (`idempotency_scope`) and Ray namespace
  when they should share, list, or reconnect to the same jobs; by default the
  registry actor name is derived from the job namespace, so pass an explicit
  `registry_actor_name` only when clients should intentionally share one registry
  actor across scopes;
- choose an analytics store that Ray workers can reach and that remains available
  after the submitting process exits;
- keep registry records small; store large outputs, reports, artifacts, datasets,
  and models in the analytics store or another storage system;
- pass large inputs by URI, lightweight handle, Ray object reference, or another
  storage reference instead of embedding them in submitted Python objects;
- make capability code and analytics-store writes safe to repeat, because retries,
  duplicate submissions, cancellation races, or worker crashes can happen;
- tune timeout, heartbeat, retention, placement, and cleanup settings for
  long-running shared clusters or KubeRay deployments.

If job history must survive RayCluster deletion or recreation, add an external
database or object store as the long-lived source of truth. Neither `ray` nor
`ray-simple` is intended to be a high-QPS queue for millisecond-scale tasks.

## Core assumptions

### 1. Keep registry records small

The registry is a job lookup table that lives in the Ray cluster. It is not a
place to store outputs, datasets, models, logs, reports, metrics tables, or
artifacts.

Keep each registry record small. It should only contain information such as:

- job id;
- job namespace and run key;
- status and timestamps;
- controller actor name and owner token;
- short error text;
- a small `CapabilityRunRef` for completed jobs.

Large data should go in the analytics store or another artifact store. The
registry should only keep references to that data.

### 2. Send workers small data they can read

Ray must send the capability and its arguments to a worker process. Those Python
objects must be serializable by Ray. Very large Python objects can be slow,
fragile, or impossible to send this way.

For large data, prefer:

- dataset, model, report, or artifact URIs;
- object storage locations;
- Ray object references for live cluster data;
- other small handles that the worker can use to load the data.

Completed jobs write their run data to the analytics store. Calling
`job.result()` returns a small `CapabilityRunRef`, not the full run payload.

### 3. Workers must be able to reach the analytics store

The analytics store configured by the client is also used by Ray workers. A job
can only write useful results if the worker can reach that store.

Make sure:

- workers have the needed URI, network access, and credentials;
- the store location remains available after the notebook or driver exits;
- completed `CapabilityRunRef.store_uri` values can be read by later clients;
- writes are safe to repeat, because retries or duplicate submissions can happen.

The registry does not store the full completed run.

### 4. The job namespace controls sharing

`idempotency_scope` is required. Think of it as the job namespace for a
workspace, project, tenant, or experiment.

Clients share and reconnect to the same jobs only when they use the same:

- `idempotency_scope`;
- registry actor name;
- registry namespace;
- compatible job backend code.

Use different scopes or registry actors for clients that should not share jobs.
When the same logical run is submitted again, the job backend is expected to return
the existing active or completed job instead of starting duplicate work.

The run key is based on the logical inputs to the capability. If an input changes
the result, it must be included in the capability config or in the
model/dataset/metric metadata used by `compute_uid(...)`. Worker resources, Ray
address, runtime environment, analytics-store URI, and `report_threshold` are not
part of that key.

### 5. Ray actor state is not permanent storage

The registry actor and controller actors can keep running after a notebook or
client process exits. That is what makes reconnecting possible.

They are still Ray processes, not permanent storage. Job tracking information is
lost if:

- the Ray cluster is shut down;
- the registry actor is lost;
- the Ray namespace is deleted;
- incompatible code is deployed;
- the actors are killed or cleaned up.

If the registry actor is lost, job metadata and duplicate-submission tracking are
lost, even if a worker already wrote rows or artifacts to the analytics store.

### 6. Clients in the same Ray namespace are trusted

The registry checks controller names and owner tokens to avoid normal race
conditions and stale updates. This is not a full security boundary.

Code that can access the same Ray namespace and registry actor is trusted. Do not
expose registry or controller actor methods to untrusted code.

### 7. Finished job state is final

Once the registry records `COMPLETED`, `FAILED`, or `CANCELLED`, that result is
final and shared by all clients.

A controller may see that a job finished before it successfully writes that final
state to the registry. If the controller dies in that gap, the registry may later
mark the job as failed. A `COMPLETED` job must include a valid, small
`CapabilityRunRef`; otherwise the public job status is treated as failed.

### 8. Status and cancellation are best effort

`RayJob.status` is a lightweight polling API. It can return old information if a
Ray call fails or times out.

Status is intentionally simple:

- `SUBMITTING` is shown as `PENDING`;
- `RUNNING` and internal `CANCELLING` are shown as `RUNNING`;
- `PENDING` can mean a normal submit handoff or an abandoned submit that is
  waiting to expire.

Use `wait()` or `result()` when you need to wait for a final answer.

`Job.cancel()` returning `True` means a cancel request was sent. It does not
promise that the final state will be `CANCELLED`. If the job finishes at the
same time as the cancel request, the final state may be `COMPLETED`, `FAILED`,
or `CANCELLED`. Timeouts passed to `wait()` or `result()` do not cancel the job.

### 9. Capability code may run more than once

The job backend tries to avoid duplicate submissions within a scope, but it does not
guarantee exactly-once execution. Ray retries, user retries, cancellation races,
and worker crashes can repeat work or leave partial side effects.

Capability code, analytics-store writes, and any other user-visible side effects
should be safe to repeat. Use unique run ids, transactions, or another workflow
specific guard when needed.

### 10. This design is for coarse jobs

The job backend creates one controller actor for each submitted job and uses one
registry actor for job lookup. This is a good fit for capability jobs that run
for minutes or hours and for moderate job-management traffic.

It is not a good fit for high-QPS millisecond tasks or thousands of tiny jobs.
Those workloads need a different design, such as pooled controllers, sharded
registries, batched task tracking, or another purpose-built coordinator.

Keep registry records small, use list limits, and store large data outside the
registry.

For Kubernetes and KubeRay deployment guidance, see the job-submission worker
environments documentation.
