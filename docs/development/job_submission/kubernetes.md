# Kubernetes and KubeRay deployment notes

This page collects Kubernetes-specific guidance for running `checkmaite` job
submission on KubeRay. General worker image and Ray `runtime_env` guidance lives
in [Worker environments](worker_environments.md).

The exact cluster YAML belongs to the platform repository, not to `checkmaite`.
The important point is that the **platform image and RayCluster definition live
outside `checkmaite`**, while the job backend connects to that cluster and
submits work into it.

## KubeRay-style deployment model

A typical RayCluster separates the head pod from one or more worker groups:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
spec:
  headGroupSpec:
    template:
      spec:
        containers:
          - name: ray-head
            image: registry.example.com/checkmaite-ray:2026-04-14
  workerGroupSpecs:
    - groupName: cpu-workers
      replicas: 2
      template:
        spec:
          containers:
            - name: ray-worker
              image: registry.example.com/checkmaite-ray:2026-04-14
              resources:
                limits:
                  cpu: "8"
                  memory: "32Gi"
    - groupName: gpu-workers
      replicas: 1
      template:
        spec:
          containers:
            - name: ray-worker
              image: registry.example.com/checkmaite-ray-gpu:2026-04-14
              resources:
                limits:
                  nvidia.com/gpu: "1"
                  cpu: "8"
                  memory: "64Gi"
```

Use deployment-specific images, resources, node selectors, tolerations, secrets,
and autoscaling settings in your platform configuration.

## Detached actors and scale-down

The default `ray` backend uses a detached registry actor and detached per-job
controller actors. Detached actors can remain alive after the submitting notebook
or driver exits, so they affect scheduling, autoscaling, pod placement, and
cleanup.

A pod that hosts a retained terminal controller actor generally cannot scale down
until that actor is killed or forgotten. The registry actor is intentionally
long-lived; if it lands on an autoscaled worker pod, that pod may stay alive for
as long as the registry exists.

For Kubernetes deployments, consider aggressive terminal-controller cleanup when
reattach-through-controller is not needed after terminal state is committed, for
example `controller_retention_s=0.0` and
`max_retained_terminal_controllers=0`. Submit-triggered sweeps are not enough for
reliable idle scale-down if the cluster becomes quiet after jobs finish.

## Head node placement

Ray head pods have extra memory and control-plane pressure from GCS, dashboard,
and cluster services. Unless intentionally using the head for lightweight
control-plane actors, configure the Ray head with `num-cpus: "0"` so nonzero-CPU
user tasks and actors do not land there.

Controller actors should normally reserve a small nonzero CPU amount, such as
the default `controller_num_cpus=0.01`, or use a custom placement resource. Avoid
`controller_num_cpus=0.0` in production Kubernetes unless placement is otherwise
controlled.

A clean production layout is often a small dedicated control-plane worker group
for the registry actor, while normal worker groups run controller actors and
capability tasks.

## Registry actor placement and resources

The registry actor should have an explicit small resource reservation or custom
resource placement in production KubeRay deployments. Use `registry_num_cpus`,
`registry_memory`, and `registry_resources` to make placement explicit.

For example, a custom resource such as `{"checkmaite-control-plane": 1}` can
force the registry onto a dedicated control-plane worker group. Avoid allowing
the registry to land on arbitrary autoscaled worker groups if that would prevent
scale-down.

The registry remains a single serialized coordination point, so keep records
small and list operations bounded. `registry_max_pending_calls` defaults to
`1024` to cap queued registry calls so many notebooks or clients do not build an
unbounded actor-call queue. `controller_max_pending_calls` defaults to `64` and
similarly caps queued calls on per-job controller actors. If those queues fill,
client calls raise `BackpressureError`; retry with exponential backoff and jitter
or tune the limits for your expected burst size. Passing `None` opts back into
Ray's unbounded pending-call behavior.

## Durability boundary and workload fit

Detached actors survive notebook or driver termination, but they do not survive
RayCluster deletion, full cluster recreation, or loss of actor memory. The
in-memory registry is sufficient for reattach across client restarts while the
RayCluster is alive. It is not durable job history across RayCluster replacement.

A production Kubernetes architecture that needs durable history should treat
detached actors as the live control plane, an external database or object store
as durable truth, and the Ray object store as transient data plane.

The per-job detached-controller design is intended for long-running capability
jobs. Workloads with thousands of concurrent jobs or very short tasks may need a
future design with a supervisor actor, controller pool, sharded registries,
batched task tracking, or DB-backed coordination.
