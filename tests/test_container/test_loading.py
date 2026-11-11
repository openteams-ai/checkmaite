from pathlib import Path

import pytest

from checkmaite_container._loading import instantiate_object, load_capability, load_symbol
from checkmaite_container._plan import ObjectSpec, RunPlanConfigurationError


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


def test_load_capability_resolves_builtins_by_task_name_or_alias(tmp_path: Path) -> None:
    from checkmaite.core import multiobject_tracking, object_detection

    assert load_capability("builtin:object_detection.MaiteEvaluation", tmp_path) is object_detection.MaiteEvaluation
    for reference in ("builtin:multiobject_tracking.MaiteEvaluation", "builtin:mot.MaiteEvaluation"):
        assert load_capability(reference, tmp_path) is multiobject_tracking.MaiteEvaluation


@pytest.mark.parametrize("reference", ["builtin:MaiteEvaluation", "builtin:semantic_segmentation.MaiteEvaluation"])
def test_load_capability_requires_a_registered_task_in_builtins(reference: str, tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="does not name a registered evaluation task.*object_detection"):
        load_capability(reference, tmp_path)


def test_load_capability_reports_unknown_builtin(tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError, match="does not export"):
        load_capability("builtin:object_detection.MissingCapability", tmp_path)


@pytest.mark.parametrize(
    ("reference", "suggestion"),
    [
        ("builtin:MaiteEvaluation", "'builtin:object_detection.MaiteEvaluation'"),
        ("builtin:object_detection", "'builtin:object_detection.Name'"),
        ("builtin:od", "'builtin:object_detection.Name'"),
    ],
)
def test_load_capability_suggests_the_explicit_builtin_form(reference: str, suggestion: str, tmp_path: Path) -> None:
    with pytest.raises(RunPlanConfigurationError) as error:
        load_capability(reference, tmp_path)
    assert suggestion in str(error.value)
