import numpy as np

from checkmaite.core._common.dataeval_bias_capability import (
    DataevalBiasConfig,
    DataevalBiasCoverageOutputs,
    DataevalBiasOutputs,
    DataevalBiasRun,
)


def test_dataeval_bias_collect_md_report():
    # minimal coverage output
    coverage = DataevalBiasCoverageOutputs(
        total=4,
        uncovered_indices=np.array([0]),
        critical_value_radii=np.array([0.1]),
        coverage_radius=0.1,
        image=None,
    )

    outputs = DataevalBiasOutputs(coverage=coverage)

    run = DataevalBiasRun(
        capability_id="test.bias",
        config=DataevalBiasConfig(),
        dataset_metadata=[{"id": "ds"}],
        model_metadata=[],
        metric_metadata=[],
        outputs=outputs,
    )

    md = run.collect_md_report(threshold=0.5)
    assert isinstance(md, str)
    assert "Bias Analysis Report" in md
