"""Check that MAITE components keep their protocol promises at runtime.

:func:`check_components` is meant to be run once, when a new model, dataset,
metric or augmentation is wired up. It is separate from
:mod:`checkmaite.core.cached_tasks`: it reads no cache, writes no cache and
returns no results.
"""

import copy
import importlib
import warnings
from collections.abc import Callable
from typing import Any

from checkmaite.core.evaluation_tasks import EvaluationTask, get_task

__all__ = ["check_components"]

# No install command: adding an extra with pip after the fact can change packages that
# uv, Poetry or conda manage, so point users at the tool they installed CheckMAITE with.
_INSTALL_HINT = (
    "Please install CheckMAITE with the spotcheck extra (checkmaite[spotcheck]), using the same "
    "tool you installed CheckMAITE with. See the install guide."
)


def check_components(
    *,
    task: str | EvaluationTask,
    model: Any = None,
    dataset: Any = None,
    metric: Any = None,
    augmentation: Any = None,
    batches: int = 2,
    batch_size: int = 2,
) -> list[str]:
    """Run components on a few real batches under MAITE's runtime checks.

    Each component is wrapped in a :mod:`maite.spotcheck` proxy for the
    protocol of the same name in ``maite.protocols.<task>``. The first
    ``batches * batch_size`` dataset items are then pushed through the same
    steps an evaluation takes: the augmentation (if given) on each batch, the
    model on the batch inputs, and ``metric.update`` with the model's
    predictions and the batch targets, followed by ``metric.compute``. Without
    a model, the metric is updated with the targets as predictions.

    The augmentation and metric are checked as deep copies, so their random
    state and accumulated results are left as they were. If one cannot be
    copied, it is checked itself with a warning: an augmentation's random state
    may advance, and a metric is reset before and after the check, also when
    the check fails. The model and dataset are not copied: the model runs on
    the items read, and a dataset must return fresh inputs on each read, as
    MAITE expects.

    A dataset is always needed, because the other components are checked on
    the values they return for real data. A dataset on its own is checked by
    reading the items.

    Parameters
    ----------
    task
        Evaluation task name, alias or :class:`EvaluationTask`. It picks the
        MAITE protocols to check against.
    model, dataset, metric, augmentation
        The components to check. Leave out the ones you do not have.
    batches, batch_size
        How many batches to run, and how many dataset items go in each. Both
        default to 2, the smallest values that cover more than one case while
        reading only 4 items: a batch of 2 catches code that only works on a
        single item, such as one that drops or squeezes the batch dimension,
        and 2 batches call each component, including ``metric.update``, more
        than once, catching code that changes behaviour after its first call.

    Returns
    -------
    list[str]
        Names of the components that were checked.

    Raises
    ------
    maite.spotcheck.SpotcheckError
        If a component breaks its protocol. A note on the error names the
        component.
    ImportError
        If MAITE's spotcheck support (MAITE 0.10 and the ``spotcheck`` extra)
        is not installed.
    KeyError
        If ``task`` is not a registered evaluation task.
    ValueError
        If MAITE has no protocols for the task, nothing is given to check, no
        dataset is given, or the dataset is shorter than
        ``batches * batch_size``.
    """
    resolved = get_task(task) if isinstance(task, str) else task
    # Check copies of the augmentation and metric, so their random state and accumulated results are untouched.
    augmentation, _ = _copy_for_check("augmentation", augmentation)
    metric, metric_copied = _copy_for_check("metric", metric)
    components = {"dataset": dataset, "augmentation": augmentation, "model": model, "metric": metric}
    given = [name for name, component in components.items() if component is not None]
    needed = _check_request(given, dataset, batches, batch_size)
    spotcheck_error, spotcheck_component = _import_spotcheck()
    try:
        protocols = importlib.import_module(f"maite.protocols.{resolved.name}")
    except ModuleNotFoundError:
        raise ValueError(f"MAITE defines no protocols for task {resolved.name!r}, so it cannot be checked.") from None

    def checked(name: str, step: Any) -> Any:
        try:
            return step()
        except spotcheck_error as error:
            error.add_note(f"The {name} broke the MAITE {resolved.name} {name.capitalize()} protocol.")
            raise

    wrapped = {
        name: checked(name, lambda c=component, n=name: spotcheck_component(getattr(protocols, n.capitalize()))(c))
        for name, component in components.items()
        if component is not None
    }
    data = wrapped["dataset"]
    augment, run_model, run_metric = (wrapped.get(name) for name in ("augmentation", "model", "metric"))

    if run_metric is not None:
        checked("metric", run_metric.reset)
    try:
        _run_batches(checked, data, augment, run_model, run_metric, needed, batch_size)
    except BaseException as error:
        if metric is not None and not metric_copied:
            _reset_after_failed_check(metric, error)
        raise
    if run_metric is not None and not metric_copied:
        # The metric itself was checked, so clear what the check added to it.
        checked("metric", run_metric.reset)
    return given


def _run_batches(
    checked: Callable[[str, Any], Any],
    data: Any,
    augment: Any,
    run_model: Any,
    run_metric: Any,
    needed: int,
    batch_size: int,
) -> None:
    """Push ``needed`` dataset items through the checked components, ``batch_size`` at a time."""
    for start in range(0, needed, batch_size):
        items = [checked("dataset", lambda i=i: data[i]) for i in range(start, start + batch_size)]
        batch = tuple(list(column) for column in zip(*items, strict=True))
        if augment is not None:
            batch = checked("augmentation", lambda b=batch: augment(b))
        inputs, targets, metadata = batch
        predictions = targets if run_model is None else checked("model", lambda x=inputs: run_model(x))
        if run_metric is not None:
            checked("metric", lambda p=predictions, t=targets, m=metadata: run_metric.update(p, t, m))
    if run_metric is not None:
        checked("metric", run_metric.compute)


def _reset_after_failed_check(metric: Any, error: BaseException) -> None:
    """Clear what a failed check added to the user's own metric, keeping ``error`` as the error raised."""
    try:
        # The unwrapped metric, so a protocol violation in reset cannot replace the original error.
        metric.reset()
    except Exception as reset_error:  # noqa: BLE001 - the check's own error is the one to report
        error.add_note(
            f"check_components also could not reset the metric after the failure ({reset_error!r}), "
            "so it may still hold results from the check."
        )


def _copy_for_check(name: str, component: Any) -> tuple[Any, bool]:
    """Return a deep copy of ``component`` and True, or the component and False if it cannot be copied."""
    if component is None:
        return None, False
    try:
        return copy.deepcopy(component), True
    except Exception as error:  # noqa: BLE001 - e.g. an object holding a lock, file handle or device resource
        warnings.warn(
            f"check_components could not copy the {name} ({error}), so it checks the {name} itself"
            + (": its random state may advance." if name == "augmentation" else " and resets it."),
            stacklevel=3,
        )
        return component, False


def _check_request(given: list[str], dataset: Any, batches: int, batch_size: int) -> int:
    """Validate the arguments and return how many dataset items the check reads."""
    if not given:
        raise ValueError("Nothing to check: pass at least a dataset.")
    if dataset is None:
        raise ValueError(f"check_components needs a dataset to run the {', '.join(given)} on.")
    if batches < 1 or batch_size < 1:
        raise ValueError("batches and batch_size must both be at least 1.")
    needed = batches * batch_size
    if len(dataset) < needed:
        raise ValueError(
            f"The dataset has {len(dataset)} items, but {batches} batches of {batch_size} need {needed}. "
            "Lower batches or batch_size."
        )
    return needed


def _import_spotcheck() -> tuple[type[Exception], Any]:
    try:
        with warnings.catch_warnings():
            # maite.spotcheck warns instead of raising when beartype is missing.
            warnings.simplefilter("ignore")
            from maite.spotcheck import SpotcheckError, spotcheck_component
    except ImportError as error:
        raise ImportError(
            f"check_components needs MAITE's spotcheck support, which is not installed.\n{_INSTALL_HINT}"
        ) from error
    return SpotcheckError, spotcheck_component
