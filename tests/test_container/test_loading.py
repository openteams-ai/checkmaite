from pathlib import Path

import pytest

from checkmaite_container._loading import instantiate_object, load_capability, load_symbol
from checkmaite_container._plan import ObjectSpec, RunPlan, RunPlanConfigurationError, TaskSpec


def test_load_symbol_supports_dotted_and_file_references(tmp_path: Path, plugin_file: Path) -> None:
    assert load_symbol("pathlib.Path", tmp_path) is Path
    fake_model = load_symbol("file:plugin.py:FakeModel", tmp_path)
    assert fake_model.__name__ == "FakeModel"
    assert load_symbol("file:plugin.py:FakeModel", tmp_path) is fake_model


def test_load_symbol_rejects_invalid_references(tmp_path: Path, plugin_file: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="dotted path"):
        load_symbol("MissingDot", tmp_path)
    with pytest.raises(RunPlanConfigurationError, match="file references"):
        load_symbol("file:plugin.py", tmp_path)
    with pytest.raises(RunPlanConfigurationError, match="does not exist"):
        load_symbol("file:missing.py:Thing", tmp_path)
    with pytest.raises(RunPlanConfigurationError, match="does not export"):
        load_symbol("file:plugin.py:Missing", tmp_path)
    with pytest.raises(RunPlanConfigurationError, match="does not export"):
        load_symbol("pathlib.Missing", tmp_path)
    with pytest.raises(RunPlanConfigurationError, match="cannot import"):
        load_symbol("missing_package.Thing", tmp_path)


def test_instantiate_object_injects_device_and_nested_objects(tmp_path: Path, plugin_file: Path) -> None:
    model = instantiate_object(
        ObjectSpec.model_validate({"class": "file:plugin.py:FakeModel", "args": {"model_id": "candidate"}}),
        plan_root=tmp_path,
        device="cuda:2",
        inject_device=True,
    )
    assert model.device == "cuda:2"
    assert model.metadata["id"] == "candidate"

    wrapper = instantiate_object(
        ObjectSpec.model_validate(
            {
                "class": "file:plugin.py:FakeWrapper",
                "args": {
                    "child": {
                        "_class": "file:plugin.py:FakeMetric",
                        "args": {"metric_id": "nested"},
                    }
                },
            }
        ),
        plan_root=tmp_path,
        device="cpu",
        inject_device=False,
    )
    assert wrapper.child.metadata["id"] == "nested"


def test_instantiate_object_keeps_class_key_in_arguments(tmp_path: Path, plugin_file: Path) -> None:
    wrapper = instantiate_object(
        ObjectSpec.model_validate({"class": "file:plugin.py:FakeWrapper", "args": {"child": {"class": "person"}}}),
        plan_root=tmp_path,
        device="cpu",
        inject_device=False,
    )
    assert wrapper.child == {"class": "person"}


def test_instantiate_object_does_not_inject_device_into_nested_objects(tmp_path: Path, plugin_file: Path) -> None:
    wrapper = instantiate_object(
        ObjectSpec.model_validate(
            {"class": "file:plugin.py:FakeWrapper", "args": {"child": {"_class": "file:plugin.py:FakeModel"}}}
        ),
        plan_root=tmp_path,
        device="cuda:0",
        inject_device=True,
    )
    assert wrapper.child.device == "missing"


def test_instantiate_object_rejects_non_callable(tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="not callable"):
        instantiate_object(
            ObjectSpec.model_validate({"class": "math.pi"}),
            plan_root=tmp_path,
            device="cpu",
            inject_device=False,
        )


def test_instantiate_object_rejects_unknown_arguments_before_calling(tmp_path: Path, plugin_file: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="invalid arguments for file:plugin.py:FakeMetric"):
        instantiate_object(
            ObjectSpec.model_validate({"class": "file:plugin.py:FakeMetric", "args": {"unknown": 1}}),
            plan_root=tmp_path,
            device="cpu",
            inject_device=False,
        )


def _plan(task: TaskSpec, **objects: dict[str, dict[str, str]]) -> RunPlan:
    return RunPlan.model_validate({"version": 1, "tasks": [task.model_dump(by_alias=True)], **objects})


_OD_DATASET = {"class": "checkmaite.core.object_detection.dataset_loaders.load_coco_detection_dataset"}
_IC_DATASET = {"class": "checkmaite.core.image_classification.dataset_loaders.load_yolo_classification_dataset"}


def test_load_capability_resolves_explicit_and_inferred_builtins(tmp_path: Path) -> None:
    explicit_task = TaskSpec(name="explicit", capability="unused")
    explicit = load_capability(
        "builtin:object_detection.MaiteEvaluation",
        explicit_task,
        _plan(explicit_task),
        tmp_path,
    )
    inferred_task = TaskSpec(name="inferred", capability="unused", dataset="data")
    inferred = load_capability(
        "builtin:MaiteEvaluation",
        inferred_task,
        _plan(inferred_task, datasets={"data": _OD_DATASET}),
        tmp_path,
    )
    assert inferred is explicit


def test_load_capability_looks_up_each_kind_by_its_own_name(tmp_path: Path) -> None:
    # An unused model that shares the dataset's name must not decide the problem type.
    task = TaskSpec(name="task", capability="unused", dataset="shared")
    plan = _plan(task, datasets={"shared": _OD_DATASET}, models={"shared": _IC_DATASET})

    capability = load_capability("builtin:MaiteEvaluation", task, plan, tmp_path)

    assert capability.__module__.startswith("checkmaite.core.object_detection")


def test_load_capability_requires_one_problem_type(tmp_path: Path) -> None:
    task = TaskSpec(name="task", capability="unused")
    with pytest.raises(RunPlanConfigurationError, match="cannot infer"):
        load_capability("builtin:MaiteEvaluation", task, _plan(task), tmp_path)

    mixed_task = TaskSpec(name="mixed", capability="unused", dataset="od", model="ic")
    plan = _plan(mixed_task, datasets={"od": _OD_DATASET}, models={"ic": _IC_DATASET})
    with pytest.raises(RunPlanConfigurationError, match="mixes objects"):
        load_capability("builtin:MaiteEvaluation", mixed_task, plan, tmp_path)


def test_load_capability_rejects_unsupported_problem_type(tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="cannot infer"):
        load_capability(
            "builtin:multiobject_tracking.MaiteEvaluation",
            TaskSpec(name="task", capability="unused"),
            _plan(TaskSpec(name="task", capability="unused")),
            tmp_path,
        )


def test_load_capability_reports_unknown_builtin(tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="does not export"):
        load_capability(
            "builtin:object_detection.MissingCapability",
            TaskSpec(name="task", capability="unused"),
            _plan(TaskSpec(name="task", capability="unused")),
            tmp_path,
        )
