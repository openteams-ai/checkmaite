# Ray Simple Backend

`ray-simple` is the lightweight Ray backend for `checkmaite` jobs:

```python
from checkmaite.jobs import configure_backend

configure_backend(
    "ray-simple",
    analytics_store={"backend": "parquet", "uri": "./job-results"},
)
```

It submits one Ray task for each capability run and returns a local `RaySimpleJob`
handle. It is intentionally much simpler than the default `ray` backend.

Use it when you want something easy to understand, easy to debug, and good enough
for local development, demos, notebooks, or simple single-driver workflows. Do
not use it as a durable shared job service.

> [!IMPORTANT]
> `ray-simple` trades durability for simplicity. It is often the easiest backend
> to use for demos and local notebooks, but the submitting client is responsible
> for duplicate-submission policy, crash recovery, and keeping job handles alive.
> If those responsibilities are not acceptable, use the default `ray` backend.

## Practical guidance

For demos and local notebooks, `ray-simple` is often the easiest backend to start
with. It has fewer moving parts than the default `ray` backend because it does
not create a shared registry actor or per-job controller actors. That makes it
easier to debug basic worker execution, analytics-store writes, and job result
handling.

The tradeoff is that operational responsibilities move to the user:

- you decide whether duplicate submissions are safe;
- you decide how to recover after a client crash;
- you keep the submitting process alive while using lifecycle APIs;
- you choose durable storage for completed run data;
- you avoid sending very large Python objects through Ray serialization when a
  URI, object-store reference, or other external storage reference would be more
  appropriate.

If those tradeoffs are unacceptable, use the default `ray` backend instead.

## When to use `ray-simple`

`ray-simple` is a good fit when:

- one Python process or notebook submits and watches the jobs;
- losing job handles after the client exits is acceptable;
- duplicate submissions are acceptable or are handled by your own code;
- you want less backend machinery while developing or debugging;
- you are running demos or small experiments where operational simplicity matters
  more than durability.

Prefer the default `ray` backend when:

- jobs must survive notebook or driver restarts;
- another client must list, cancel, or reconnect to existing jobs;
- duplicate submissions must be suppressed by the backend;
- multiple users or processes share the same Ray cluster;
- you need production-style job tracking on KubeRay or a long-running Ray cluster.

## Core assumptions

### 1. Job tracking lives only in the submitting process

`ray-simple` keeps submitted jobs in an in-memory dictionary on the
`RaySimpleBackend` instance.

That means:

- `list_jobs()` only lists jobs submitted through that backend object;
- `get_job(job_id)` only works for jobs still remembered by that backend object;
- if the notebook, driver, or Python process exits, the job handles are lost;
- a new client cannot recreate a `RaySimpleJob` from an old job id.

The Ray task may or may not continue after the driver exits, depending on Ray
runtime behavior, but `ray-simple` will no longer have a handle that can observe,
wait for, or cancel it.

### 2. Users are responsible for job de-duplication

Every call to `submit_capability(...)` creates a new Ray task.

`ray-simple` does not provide:

- cross-client duplicate detection;
- an `idempotency_scope`;
- a shared run-key registry;
- suppression of repeated submissions for the same logical capability run.

If duplicate work or duplicate side effects matter, handle that in your workflow,
your capability code, or the storage layer. Capability runs and analytics-store
writes should be idempotent when duplicate submissions are possible.

### 3. Users are responsible for crash recovery

If the submitting client crashes or exits, `ray-simple` does not provide a way to
recover the local job handles.

If you need to recover from a client crash, use one of these patterns instead:

- use the default `ray` backend, which is designed around shared job metadata;
- store your own application-level submission records;
- design capabilities so repeating a submission is safe;
- write outputs to durable external storage and treat the analytics store as the
  source of completed-run data.

### 4. Avoid serializing large Python objects

Completed jobs write their run data to the configured analytics store. Calling
`job.result()` returns a small `CapabilityRunRef`, not the full capability run or
large output payloads.

Ray still has to serialize the capability object and submission arguments that
are sent to the worker task. Avoid placing large datasets, model objects, or
large output payloads directly in those Python objects when a reference would be
more appropriate.

Prefer passing large data through:

- dataset or model URIs;
- object storage locations;
- Ray object-store references;
- other application-level storage references.

Use the analytics store for completed run records and use external storage for
large artifacts.

### 5. Status and cancellation are best-effort Ray observations

`ray-simple` maps Ray task state into the `checkmaite.jobs.JobStatus` protocol:

- `status` uses `ray.wait(..., timeout=0)` and, when ready, a bounded
  `ray.get(..., timeout=0)`;
- Ray Core object readiness does not distinguish queued work from executing work;
- as a local polling heuristic, the first non-ready observation is reported as
  `PENDING` and later non-ready observations are reported as `RUNNING`;
- terminal Ray success maps to `COMPLETED`;
- task cancellation maps to `CANCELLED`;
- task errors map to `FAILED`.

`cancel()` calls `ray.cancel(...)` and treats the local handle as cancelled once
the cancellation request is issued. Ray cancellation is best-effort, so user code
or external side effects may already have run.

Timeouts passed to `result()` or `wait()` do not cancel the Ray task. They only
bound how long the caller waits.

### 6. Ray runtime lifecycle is shared with the process

`RaySimpleBackend` initializes Ray when needed. Its shutdown behavior is simple:

- `shutdown(wait=True)` waits for known jobs and then calls `ray.shutdown()`;
- `shutdown(wait=False)` returns immediately and does not shut down Ray.

If other code in the same process shares the Ray runtime, coordinate shutdown
carefully.
