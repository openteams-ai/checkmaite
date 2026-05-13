import pytest
import torch

from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationConfig,
    MaiteEvaluationOutputs,
    MaiteEvaluationRun,
)
from checkmaite.core.image_classification.maite_evaluation_capability import MaiteEvaluation
from checkmaite.core.report._gradient import HAS_GRADIENT


@pytest.fixture
def test_run_ic(fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default):
    capability = MaiteEvaluation()
    output = capability.run(
        datasets=[fake_ic_dataset_default], metrics=[fake_ic_metric_default], models=[fake_ic_model_default]
    )
    assert output.model_dump()  # smoke test

    return output


def test_collect_md_report_ic(test_run_ic):
    assert test_run_ic.collect_md_report(threshold=0.5)  # smoke test


def test_maite_evaluation_extract_preserves_overall_and_class_metrics() -> None:
    run = MaiteEvaluationRun(
        capability_id="maite-evaluation",
        config=MaiteEvaluationConfig(),
        dataset_metadata=[{"id": "dataset"}],
        model_metadata=[{"id": "model"}],
        metric_metadata=[{"id": "accuracy"}],
        outputs=MaiteEvaluationOutputs(
            overall_metric_name="accuracy",
            result={"accuracy": 0.75},
            class_metrics={"cat": 0.5, "dog": None},
        ),
    )

    records = run.extract()

    assert {record.output_value for record in records} == {0.75, 0.5}


def test_maite_evaluation_rejects_malformed_per_class_metric(
    fake_ic_model_default, fake_ic_dataset_default, fake_ic_metric_default
) -> None:
    metric = type(fake_ic_metric_default)(
        calculated_metrics={
            "per_class_flag": torch.tensor(1.0),
            "fake_metric": torch.tensor(0.75),
            "unexpected": torch.tensor(0.1),
        },
        return_key="fake_metric",
    )

    with pytest.raises(RuntimeError, match="single value"):
        MaiteEvaluation().run(datasets=[fake_ic_dataset_default], models=[fake_ic_model_default], metrics=[metric])


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_collect_report_consumables_ic(test_run_ic):
    assert test_run_ic.collect_report_consumables(threshold=0.5)  # smoke test
