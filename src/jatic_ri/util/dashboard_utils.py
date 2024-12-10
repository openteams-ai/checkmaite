"""Utility functions to be used with dashboards"""

from __future__ import annotations

from typing import Any

from jatic_ri.image_classification.test_stages.impls.baseline_evaluation import (
    BaselineEvaluation as BaselineEvaluationIC,
)
from jatic_ri.image_classification.test_stages.impls.dataeval_bias_test_stage import (
    DatasetBiasTestStage as DatasetBiasTestStageIC,
)
from jatic_ri.image_classification.test_stages.impls.dataeval_feasibility_test_stage import (
    DatasetFeasibilityTestStage as DatasetFeasibilityTestStageIC,
)
from jatic_ri.image_classification.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTestStage as DatasetLintingTestStageIC,
)
from jatic_ri.image_classification.test_stages.impls.dataeval_shift_test_stage import (
    DatasetShiftTestStage as DatasetShiftTestStageIC,
)
from jatic_ri.image_classification.test_stages.impls.nrtk_test_stage import NRTKTestStage as NRTKTestStageIC
from jatic_ri.image_classification.test_stages.impls.survivor_test_stage import SurvivorTestStage as SurvivorTestStageIC
from jatic_ri.image_classification.test_stages.impls.xaitk_test_stage import XAITKTestStage as XAITKTestStageIC
from jatic_ri.object_detection.test_stages.impls.baseline_evaluation import BaselineEvaluation as BaselineEvaluationOD
from jatic_ri.object_detection.test_stages.impls.dataeval_bias_test_stage import (
    DatasetBiasTestStage as DatasetBiasTestStageOD,
)
from jatic_ri.object_detection.test_stages.impls.dataeval_feasibility_test_stage import (
    DatasetFeasibilityTestStage as DatasetFeasibilityTestStageOD,
)
from jatic_ri.object_detection.test_stages.impls.dataeval_linting_test_stage import (
    DatasetLintingTestStage as DatasetLintingTestStageOD,
)
from jatic_ri.object_detection.test_stages.impls.dataeval_shift_test_stage import (
    DatasetShiftTestStage as DatasetShiftTestStageOD,
)
from jatic_ri.object_detection.test_stages.impls.nrtk_test_stage import NRTKTestStage as NRTKTestStageOD
from jatic_ri.object_detection.test_stages.impls.reallabel_test_stage import Config as ReallabelConfig
from jatic_ri.object_detection.test_stages.impls.reallabel_test_stage import RealLabelTestStage as RealLabelTestStageOD
from jatic_ri.object_detection.test_stages.impls.survivor_test_stage import SurvivorTestStage as SurvivorTestStageOD
from jatic_ri.object_detection.test_stages.impls.xaitk_test_stage import XAITKTestStage as XAITKTestStageOD


def rehydrate_test_stage_od(  # noqa: C901
    config: dict[str, Any],
) -> (
    BaselineEvaluationOD
    | NRTKTestStageOD
    | RealLabelTestStageOD
    | SurvivorTestStageOD
    | XAITKTestStageOD
    | DatasetShiftTestStageOD
    | DatasetLintingTestStageOD
    | DatasetBiasTestStageOD
    | DatasetFeasibilityTestStageOD
):
    """Initialize test stage object based on config dictionary"""
    if config["TYPE"] == "RealLabelTestStage":
        reallabel_config = ReallabelConfig(**config["CONFIG"])
        return RealLabelTestStageOD(config=reallabel_config)
    if config["TYPE"] == "NRTKTestStage":
        return NRTKTestStageOD(config["CONFIG"])
    if config["TYPE"] == "XAITKTestStage":
        return XAITKTestStageOD(config["CONFIG"])
    if config["TYPE"] == "SurvivorTestStage":
        return SurvivorTestStageOD(config["CONFIG"])
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return BaselineEvaluationOD()
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return DatasetFeasibilityTestStageOD()
    if config["TYPE"] == "DatasetBiasTestStage":
        return DatasetBiasTestStageOD()
    if config["TYPE"] == "DatasetLintingTestStage":
        return DatasetLintingTestStageOD()
    if config["TYPE"] == "DatasetShiftTestStage":
        return DatasetShiftTestStageOD()

    raise RuntimeError(f'Unable to instantiate TestStage object from config: {config["TYPE"]}')


def rehydrate_test_stage_ic(
    config: dict[str, Any],
) -> (
    BaselineEvaluationIC
    | NRTKTestStageIC
    | SurvivorTestStageIC
    | XAITKTestStageIC
    | DatasetShiftTestStageIC
    | DatasetLintingTestStageIC
    | DatasetBiasTestStageIC
    | DatasetFeasibilityTestStageIC
):
    """Initialize test stage object based on config dictionary"""
    if config["TYPE"] == "NRTKTestStage":
        return NRTKTestStageIC(config["CONFIG"])
    if config["TYPE"] == "XAITKTestStage":
        return XAITKTestStageIC(config["CONFIG"])
    if config["TYPE"] == "SurvivorTestStage":
        return SurvivorTestStageIC(config["CONFIG"])
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return BaselineEvaluationIC()
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return DatasetFeasibilityTestStageIC()
    if config["TYPE"] == "DatasetBiasTestStage":
        return DatasetBiasTestStageIC()
    if config["TYPE"] == "DatasetLintingTestStage":
        return DatasetLintingTestStageIC()
    if config["TYPE"] == "DatasetShiftTestStage":
        return DatasetShiftTestStageIC()

    raise RuntimeError(f'Unable to instantiate TestStage object from config: {config["TYPE"]}')


def create_download_link(path: str, label: str | None = None, download_filename: str | None = "report.html") -> str:
    """Constructs an html string which can be used as a hyperlink for
    downloading a file. Clicking the link will download the file.

    The link contains javascript code which pull the Jupyter token from
    the document cookie.

    Parameters
    ----------
    path: str
        path to file from $HOME. Note that if link will be distributed,
        the path must be accessible to others (i.e in a shared location),
        or if this is being run as a deployed app, the app user must have
        access to that location.
    label: str
        Optional. The visible text of the link. Defaults to the text of the url.
    download_filename: str
        Optional. The name of the file that is downloaded when the link is clicked. Defaults to 'report.html'

    Returns
    -------
    HTML formatted download link

    """
    if not label:
        label = path

    return f"""
    <a
        href="#"
        onclick="
        var url = location.href.split('/').splice(0, 5).join('/') + '/files/{path}';
        var xsrfTokenMatch = document.cookie.match('\\\\b_xsrf=([^;]*)\\\\b');
        var fullUrl = new window.URL(url);
        fullUrl.searchParams.append('_xsrf', xsrfTokenMatch[1]);
        var link = document.createElement('a');
        link.download = '{download_filename}';
        link.href = fullUrl.toString();
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    ">{label}</a>
    """
