from checkmaite.core._common.maite_evaluation_capability import (
    MaiteEvaluationConfig,
    MaiteEvaluationOutputs,
    MaiteEvaluationRun,
)
from checkmaite.core.report import InlineTextReport


def test_maite_collect_md_report_simple():
    outputs = MaiteEvaluationOutputs(overall_metric_name="acc", result={"acc": 0.75}, class_metrics=None)

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


def test_maite_collect_md_report_with_class_metrics():
    # create class metrics branch
    outputs = MaiteEvaluationOutputs(
        overall_metric_name="acc",
        result={"acc": 0.6, "per_class_flag": 1, "0": 0.5},
        class_metrics={"cat": None, "dog": 0.7},
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
