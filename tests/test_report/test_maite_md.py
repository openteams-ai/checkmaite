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


def test_report_records_its_provenance():
    """IR-3-S-3: a rendered report identifies the run that produced it, so it needs no metadata sidecar."""
    import checkmaite

    run = MaiteEvaluationRun(
        capability_id="test.maite",
        config=MaiteEvaluationConfig(batch_size=4),
        dataset_metadata=[{"id": "ds-a"}, {"id": "ds-b"}],
        model_metadata=[{"id": "model-1"}],
        metric_metadata=[{"id": "acc"}],
        outputs=MaiteEvaluationOutputs(metrics={}),
    )

    content = run.collect_md_report(threshold=0.5).content
    provenance = content[content.index("## Provenance") :]

    assert f"`{run.run_uid}`" in provenance
    assert "`test.maite`" in provenance
    assert f"`{checkmaite.__version__}`" in provenance
    assert "`ds-a`, `ds-b`" in provenance
    assert "`model-1`" in provenance
    assert "`acc`" in provenance
    assert '"batch_size": 4' in provenance
