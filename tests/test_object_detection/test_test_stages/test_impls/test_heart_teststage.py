import pytest

# Skip all tests in this file if heart_library isn't available
pytest.importorskip("heart_library")

# Module-level imports after importorskip to prevent collection errors
from jatic_ri.object_detection.test_stages import HeartAttackConfig, HeartTestStage  # noqa: E402
from jatic_ri.object_detection.test_stages._impls.heart_test_stage import _DEFAULT_HEART_ATTACK_PARAMETERS  # noqa: E402

ATTACK_CONFIGS = [
    HeartAttackConfig(name="PGD", strength="weak", parameters={"max_iter": 1, "eps": 1, "eps_step": 0.2}),
    HeartAttackConfig(name="PGD", strength="strong", parameters={"max_iter": 2, "eps": 2, "eps_step": 0.2}),
    HeartAttackConfig(
        name="Patch",
        strength="weak",
        parameters={
            "rotation_max": 0.0,
            "scale_min": 0.5,
            "scale_max": 1.0,
            "distortion_scale_max": 0.0,
            "learning_rate": 1.99,
            "max_iter": 1,
            "batch_size": 16,
            "patch_shape": (3, 10, 10),
            "patch_location": (2, 2),
            "patch_type": "square",
            "optimizer": "Adam",
        },
    ),
    HeartAttackConfig(
        name="Patch",
        strength="strong",
        parameters={
            "rotation_max": 0.0,
            "scale_min": 0.5,
            "scale_max": 1.0,
            "distortion_scale_max": 0.0,
            "learning_rate": 1.99,
            "max_iter": 2,
            "batch_size": 16,
            "patch_shape": (3, 100, 100),
            "patch_location": (440, 100),
            "patch_type": "square",
            "optimizer": "Adam",
        },
    ),
]


@pytest.mark.heart
@pytest.mark.unsupported
@pytest.mark.real_data
@pytest.mark.parametrize("attack_config", ATTACK_CONFIGS)
def test_run(mocker, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, attack_config):
    stage = HeartTestStage(attack_configs=[attack_config])
    stage.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    stage.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    stage.load_metric(metric=fake_od_metric_default, metric_id=fake_od_metric_default.metadata["id"])

    run = stage.run(use_stage_cache=True)

    mocker.patch.object(stage, "_run", side_effect=AssertionError("_run() called while cache hit was expected"))
    cached_run = stage.run(use_stage_cache=True)

    assert cached_run.outputs.baseline.result == run.outputs.baseline.result
    assert [o.result for o in cached_run.outputs.attacked] == [o.result for o in run.outputs.attacked]


@pytest.mark.heart
@pytest.mark.unsupported
@pytest.mark.real_data
@pytest.mark.parametrize("attack_config", ATTACK_CONFIGS)
def test_default_attack_parameters(
    mocker, fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, attack_config
):
    mocker.patch.dict(
        _DEFAULT_HEART_ATTACK_PARAMETERS,
        clear=True,
        values={(attack_config.name, attack_config.strength): attack_config.parameters},
    )

    default_attack_config = HeartAttackConfig(name=attack_config.name, strength=attack_config.strength)

    assert default_attack_config is not attack_config.parameters
    assert default_attack_config.parameters == attack_config.parameters


@pytest.mark.heart
@pytest.mark.unsupported
@pytest.mark.real_data
@pytest.mark.parametrize("attack_config", ATTACK_CONFIGS)
def test_report_consumables(
    fake_od_model_default, fake_od_dataset_default, fake_od_metric_default, attack_config
) -> None:
    stage = HeartTestStage(attack_configs=[attack_config])
    stage.load_model(model=fake_od_model_default, model_id=fake_od_model_default.metadata["id"])
    stage.load_dataset(dataset=fake_od_dataset_default, dataset_id=fake_od_dataset_default.metadata["id"])
    stage.load_metric(metric=fake_od_metric_default, metric_id=fake_od_metric_default.metadata["id"])

    stage.run()

    slides = stage.collect_report_consumables()
    for slide in slides:
        assert slide["deck"] == "object_detection_model_evaluation"
        assert slide["layout_name"] == "ThreeSection"

        args = slide["layout_arguments"]
        assert "title" in args
        assert "left_section_heading" in args
        assert "left_section_content" in args
        assert "mid_section_heading" in args
        assert "mid_section_content" in args
        assert "right_section_heading" in args
        assert "right_section_content" in args
