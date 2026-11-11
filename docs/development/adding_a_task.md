# Adding an evaluation task

An evaluation task is a kind of AI problem, such as image classification,
object detection or multi-object tracking. CheckMAITE's core does not hard-code
its tasks. Each one is described by an `EvaluationTask` in
`checkmaite.core.evaluation_tasks`, and the core looks tasks up by name when it
needs them:

- the prediction cache uses the task's `target_adapter` to store and restore
  model predictions and dataset targets;
- the evaluation loop uses `materialize_inputs` to decide whether one-shot
  input streams, such as video frames, must be turned into lists;
- the batch container resolves `builtin:<task>.Name` from the task's
  `namespace`.

A new task can therefore live in its own package. The built-in tasks are
described the same way, in `checkmaite/core/<task>/_task.py`, and each task
namespace's `__init__.py` registers its task with `register_task(TASK)`.

## 1. Describe the task

```python
# my_package/segmentation.py
from typing import Protocol

from maite.protocols import ArrayLike
from pydantic import TypeAdapter

from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask


class SegmentationTarget(Protocol):
    """What models return and datasets hold, like MAITE's task protocols."""

    @property
    def mask(self) -> ArrayLike: ...


class CachedSegmentationTarget(CapabilityOutputsBase):
    """How one prediction or target is cached."""

    mask: ArrayLike


TASK: EvaluationTask[SegmentationTarget] = EvaluationTask(
    name="semantic_segmentation",
    namespace="my_package.segmentation",
    target_adapter=TypeAdapter(CachedSegmentationTarget),
    aliases=("seg",),
    # The plugin API version this package was written and tested against.
    plugin_api_version="1.0.0",
)
```

- `name` is the task's stable identifier. Use the MAITE protocol module name
  when one exists (`maite.protocols.<name>`).
- `namespace` is the module that exports the task's capabilities.
- `target_adapter` is a pydantic `TypeAdapter` for one prediction or target.
  Values are validated with `from_attributes=True`, so any object with a
  `mask` attribute is accepted. Predictions and targets read back from the
  cache are instances of `CachedSegmentationTarget`. Annotating the task as
  `EvaluationTask[SegmentationTarget]` makes the type checker verify that
  `CachedSegmentationTarget` satisfies the protocol, and that a task object
  passed as `task=` to `evaluate` or `predict` matches the model's target
  type. Fields typed as MAITE `ArrayLike` on a `CapabilityOutputsBase` are
  stored as binary files next to the cache entry.
  An adapter accepts any type pydantic can validate, not only classes. For a
  target that is itself an array, adapt the protocol's `TargetType` directly,
  as image classification does:
  `TypeAdapter(TargetType, config=ConfigDict(arbitrary_types_allowed=True))`.
- A task has one type for both model predictions and dataset targets, as in
  MAITE, whose models return and whose datasets hold the same `TargetType`. A
  task whose predictions differ in type from its targets, such as a
  distribution predicted against a label, cannot be described yet.
- `aliases` are other accepted names, such as DataMAITE's short task codes.
- Set `materialize_inputs=True` if a datum's input can be a one-shot iterable.
- `plugin_api_version` is required for tasks loaded from an entry point. See
  [Compatible CheckMAITE versions](#compatible-checkmaite-versions).

`register_task` rejects a `target_adapter` that is not a `TypeAdapter`, or
whose type pydantic cannot build a validator for, for example because of an
unresolved forward reference.

## 2. Add capabilities

Most capabilities only need a subclass that names the task. The shared
`MaiteEvaluationBase` works for any task whose metrics follow the MAITE
protocol:

```python
from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationBase,
    MaiteEvaluationConfig,
)


class MaiteEvaluation(MaiteEvaluationBase):
    """Evaluate one segmentation model and dataset with one or more metrics."""

    task = "semantic_segmentation"

    @classmethod
    def _create_config(cls) -> MaiteEvaluationConfig:
        return MaiteEvaluationConfig()
```

Setting `task` on a capability tells the cache which task's `target_adapter` to
use. Every capability must set it, and running one that does not, or that
names an unregistered task, raises. CheckMAITE never guesses
the task, and the task must be registered (see
[Register the task](#4-register-the-task)).

## 3. Export cached tasks

`checkmaite.core.cached_tasks.evaluate` and `predict` require `task=`. Export
versions with your task bound, so your users call them exactly like
`maite.tasks.evaluate` and `maite.tasks.predict`:

```python
# my_package/segmentation.py, below TASK
from checkmaite.core import cached_tasks

_cached = cached_tasks.for_task(TASK)
evaluate = _cached.evaluate
predict = _cached.predict
evaluate_from_predictions = cached_tasks.evaluate_from_predictions
```

```python
from my_package import segmentation

results, predictions, _ = segmentation.evaluate(
    model=model, metric=metric, dataset=dataset
)
```

The bound functions take the same arguments as the module-level ones without
`task`, and their predictions are typed with the task's target protocol. The
built-in tasks export the same functions from
`checkmaite.core.<task>.cached_tasks`. `for_task` does not register the task;
calling a bound function for an unregistered task raises `ValueError`.

## 4. Register the task

Declare an entry point in the `checkmaite.tasks` group that points at the
`EvaluationTask`:

```toml
[project.entry-points."checkmaite.tasks"]
semantic_segmentation = "my_package.segmentation:TASK"
```

CheckMAITE loads these entry points the first time a task is looked up. The
outcome of each load is recorded with the other plugins and can be inspected
with `checkmaite.core._plugins.list_loaded_plugins("checkmaite.tasks")`. An
entry point that fails to import, does not resolve to an `EvaluationTask`, has
a malformed `target_adapter` or an incompatible `plugin_api_version` is recorded
as `failed` with the reason. It does not affect the other tasks.

For quick experiments, `register_task(TASK)` registers a task without
packaging it. Tasks registered this way are not version-checked.

### Compatible CheckMAITE versions

`plugin_api_version` follows the same rule as for
[capability plugins](plugins.md#api-version-compatibility): its major version
must match CheckMAITE's `PLUGIN_API_VERSION`, otherwise the task is not
registered. Write the version you built and tested against as a string, as in
the example above. Don't import `PLUGIN_API_VERSION` to set it: that always
matches the installed CheckMAITE, so the check could never fail.

The `checkmaite.tasks` entry point group is new after CheckMAITE 0.7.0. In the
package's dependencies, require a later release and cap the version at the
next minor release you have not tested, for example:

```toml
[project]
dependencies = ["checkmaite>0.7.0,<0.9"]
```

## Naming the task

The task is always named, never inferred from the data. Several tasks can
accept the same targets: image classification accepts any array, and
instance-segmentation targets with `boxes`, `labels`, `scores` and `mask` also
satisfy the object-detection target type. Naming the task decides how they are
cached, for example whether the mask is kept.

CheckMAITE checks the first batch of predictions and targets against the named
task's `target_adapter`, and raises `ValueError` if they do not match. The
check cannot catch every wrong name:
image classification accepts any array, so a segmentation model whose masks
are plain arrays still passes as image classification.

The task is part of the prediction and metric-result cache keys, so an entry
is never reused for a different task. An alias shares its task's entry. A
cache lookup never reads the dataset. Prediction and metric-result caches
written by CheckMAITE 0.7.0 and earlier are not reused; they are recomputed
once.

## Evaluating and caching a task

This example registers a task, evaluates a model with two metrics and reuses
the cached predictions for the second one:

```python
import tempfile

import numpy as np
from maite.protocols import ArrayLike
from pydantic import TypeAdapter

import checkmaite
from checkmaite.core import cached_tasks as generic_cached_tasks
from checkmaite.core.capability_core import CapabilityOutputsBase
from checkmaite.core.evaluation_tasks import EvaluationTask, register_task


class SegmentationTarget(CapabilityOutputsBase):
    mask: ArrayLike


TASK = register_task(
    EvaluationTask(
        name="semantic_segmentation",
        namespace=__name__,
        target_adapter=TypeAdapter(SegmentationTarget),
        aliases=("seg",),
    )
)
cached_tasks = generic_cached_tasks.for_task(TASK)


class Mask:
    def __init__(self, mask):
        self.mask = mask


class Dataset:
    metadata = {"id": "toy-roads"}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return np.zeros((3, 8, 8)), Mask(np.full((8, 8), index % 2)), {"id": index}


class Model:
    metadata = {"id": "all-background"}
    calls = 0

    def __call__(self, images):
        Model.calls += 1
        return [Mask(np.zeros((8, 8))) for _ in images]


class PixelAccuracy:
    def __init__(self, metric_id):
        self.metadata = {"id": metric_id}
        self.reset()

    def reset(self):
        self.scores = []

    def update(self, predictions, targets, metadata):
        for prediction, target in zip(predictions, targets):
            matches = np.asarray(prediction.mask) == np.asarray(target.mask)
            self.scores.append(float(np.mean(matches)))

    def compute(self):
        return {"pixel_accuracy": float(np.mean(self.scores))}


checkmaite.cache_path(tempfile.mkdtemp())
model, dataset = Model(), Dataset()
first, _, _ = cached_tasks.evaluate(
    model=model, metric=PixelAccuracy("accuracy"), dataset=dataset
)
second, _, _ = cached_tasks.evaluate(
    model=model, metric=PixelAccuracy("accuracy-2"), dataset=dataset
)
print(first, second, Model.calls)
# {'pixel_accuracy': 0.5} {'pixel_accuracy': 0.5} 4
```

The model runs once per image. The second metric is computed from the cached
predictions.

`strict_cache_serialization=True` only caches values it knows to be lossless.
It declines custom target objects, including values produced by the task's
`target_adapter`: the evaluation still runs, but nothing is cached and a warning
says why. Use the default compatibility serialization for such tasks.

## Checking the result

With the package from [step 1](#1-describe-the-task) installed:

```python
from checkmaite.core.evaluation_tasks import get_task, list_tasks

[task.name for task in list_tasks()]
# ['image_classification', 'multiobject_tracking', 'object_detection', 'semantic_segmentation']
get_task("seg").namespace
# 'my_package.segmentation'
```

`tests/test_core/test_evaluation_tasks.py` installs a throwaway package that
adds a segmentation task this way. The batch container then evaluates it with
two metrics, and the test checks that the second metric reuses the cached
predictions and that the reports and analytics tables are written.

## Adding capabilities to an existing task

To add capabilities to a task that already exists, rather than adding a new
task, use the [plugin system](plugins.md).
