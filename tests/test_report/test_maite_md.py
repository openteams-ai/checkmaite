from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationConfig,
    MaiteEvaluationOutputs,
    MaiteEvaluationRun,
    MaiteMetricResult,
)
from checkmaite.core.report import InlineTextReport


def test_maite_collect_md_report_simple():
    outputs = MaiteEvaluationOutputs(
        metrics={
            "acc": MaiteMetricResult(
                metric_id="acc",
                overall_metric_name="acc",
                result={"acc": 0.75},
                scalar_values={"acc": 0.75},
                class_metrics=None,
            )
        }
    )

    run = MaiteEvaluationRun(
        capability_id="test.maite",
        config=MaiteEvaluationConfig(),
        dataset_metadata=[{"id": "ds"}],
        model_metadata=[{"id": "m"}],
        metric_metadata=[],
        outputs=outputs,
    )

    report = run.collect_md_report(threshold=0.5)
    assert isinstance(report, InlineTextReport)
    assert report.media_type == "text/markdown"
    assert report.filename == "test.maite.md"
    assert "Model Evaluation Summary" in report.content


def test_legacy_gradient_slide_has_item_for_structured_metric():
    outputs = MaiteEvaluationOutputs(
        metrics={
            "structured": MaiteMetricResult(
                metric_id="structured",
                overall_metric_name=None,
                result={"details": {"count": 2}},
                scalar_values={},
                class_metrics=None,
            )
        }
    )
    run = MaiteEvaluationRun(
        capability_id="test.maite-structured",
        config=MaiteEvaluationConfig(),
        dataset_metadata=[{"id": "ds"}],
        model_metadata=[{"id": "m"}],
        metric_metadata=[{"id": "structured"}],
        outputs=outputs,
    )
    collector = type(run).collect_report_consumables.__wrapped__.__wrapped__

    slides = collector(run, threshold=0.5)

    assert "item" in slides[0]["layout_arguments"]


def test_maite_collect_md_report_with_class_metrics():
    # create class metrics branch
    outputs = MaiteEvaluationOutputs(
        metrics={
            "acc": MaiteMetricResult(
                metric_id="acc",
                overall_metric_name="acc",
                result={"acc": 0.6},
                scalar_values={"acc": 0.6},
                class_metrics={"cat": None, "dog": 0.7},
            )
        }
    )

    run = MaiteEvaluationRun(
        capability_id="test.maite2",
        config=MaiteEvaluationConfig(),
        dataset_metadata=[{"id": "ds"}],
        model_metadata=[{"id": "m", "index2label": {}}],
        metric_metadata=[],
        outputs=outputs,
    )

    report = run.collect_md_report(threshold=0.5)
    assert isinstance(report, InlineTextReport)
    assert report.media_type == "text/markdown"
    assert report.filename == "test.maite2.md"
    assert "Model Evaluation Summary" in report.content
