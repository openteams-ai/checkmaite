from collections.abc import Callable
from typing import Any

import panel as pn

from jatic_ri.core.image_classification.dataeval_bias_capability import (
    DataevalBias as DataevalBiasIC,
)
from jatic_ri.core.image_classification.dataeval_cleaning_capability import (
    DataevalCleaning as DataevalCleaningIC,
)
from jatic_ri.core.image_classification.dataeval_feasability_capability import (
    DataevalFeasibility as DataevalFeasibilityIC,
)
from jatic_ri.core.image_classification.dataeval_shift_capability import (
    DataevalShift as DataevalShiftIC,
)
from jatic_ri.core.image_classification.maite_evaluation_capability import (
    MaiteEvaluation as MaiteEvaluationIC,
)
from jatic_ri.core.image_classification.nrtk_augmentation_capability import (
    NrtkAugmentation as NrtkAugmentationIC,
)

# from jatic_ri.core.image_classification.test_stages import (
#     SurvivorTestStage as SurvivorTestStageIC,
# )
from jatic_ri.core.image_classification.xaitk_explainable_capability import (
    XaitkExplainable as XaitkExplainableIC,
)
from jatic_ri.core.object_detection.dataeval_bias_capability import (
    DataevalBias as DataevalBiasOD,
)
from jatic_ri.core.object_detection.dataeval_cleaning_capability import (
    DataevalCleaning as DataevalCleaningOD,
)
from jatic_ri.core.object_detection.dataeval_feasability_capability import (
    DataevalFeasibility as DataevalFeasibilityOD,
)
from jatic_ri.core.object_detection.dataeval_shift_capability import (
    DataevalShift as DataevalShiftOD,
)
from jatic_ri.core.object_detection.maite_evaluation_capability import MaiteEvaluation as MaiteEvaluationOD
from jatic_ri.core.object_detection.nrtk_augmentation_capability import (
    NrtkAugmentation as NrtkAugmentationOD,
)

# from jatic_ri.core.object_detection.test_stages import (
#     RealLabelConfig,
# )
# from jatic_ri.core.object_detection.test_stages import (
#     RealLabelTestStage as RealLabelTestStageOD,
# )
# from jatic_ri.core.object_detection.test_stages import (
#     SurvivorTestStage as SurvivorTestStageOD,
# )
from jatic_ri.core.object_detection.xaitk_explainable_capability import (
    XaitkExplainable as XaitkExplainableOD,
)


def rehydrate_test_stage_od(
    config: dict[str, Any],
) -> (
    MaiteEvaluationOD
    | NrtkAugmentationOD
    # | RealLabelTestStageOD
    # | SurvivorTestStageOD
    | XaitkExplainableOD
    | DataevalShiftOD
    | DataevalCleaningOD
    | DataevalBiasOD
    | DataevalFeasibilityOD
):
    """Initialize test stage object based on config dictionary"""
    # if config["TYPE"] == "RealLabelTestStage":
    #     reallabel_config = RealLabelConfig(**config["CONFIG"])
    #     return RealLabelTestStageOD(config=reallabel_config)
    if config["TYPE"] == "NRTKTestStage":
        return NrtkAugmentationOD()
    if config["TYPE"] == "XAITKTestStage":
        return XaitkExplainableOD()
    # if config["TYPE"] == "SurvivorTestStage":
    #     return SurvivorTestStageOD(config["CONFIG"])
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return MaiteEvaluationOD()
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return DataevalFeasibilityOD()
    if config["TYPE"] == "DatasetBiasTestStage":
        return DataevalBiasOD()
    if config["TYPE"] == "DatasetCleaningTestStage":
        return DataevalCleaningOD()
    if config["TYPE"] == "DatasetShiftTestStage":
        return DataevalShiftOD()

    raise RuntimeError(f'Unable to instantiate TestStage object from config: {config["TYPE"]}')


def rehydrate_test_stage_ic(
    config: dict[str, Any],
) -> (
    MaiteEvaluationIC
    | NrtkAugmentationIC
    # | SurvivorTestStageIC
    | XaitkExplainableIC
    | DataevalShiftIC
    | DataevalCleaningIC
    | DataevalBiasIC
    | DataevalFeasibilityIC
):
    """Initialize test stage object based on config dictionary"""
    if config["TYPE"] == "NRTKTestStage":
        return NrtkAugmentationIC()
    if config["TYPE"] == "XAITKTestStage":
        return XaitkExplainableIC()
    # if config["TYPE"] == "SurvivorTestStage":
    #     return SurvivorTestStageIC(config["CONFIG"])
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return MaiteEvaluationIC()
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return DataevalFeasibilityIC()
    if config["TYPE"] == "DatasetBiasTestStage":
        return DataevalBiasIC()
    if config["TYPE"] == "DatasetCleaningTestStage":
        return DataevalCleaningIC()
    if config["TYPE"] == "DatasetShiftTestStage":
        return DataevalShiftIC()

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


def _center_vertically(panel_object: pn.viewable.Viewable) -> pn.Column:
    """Create a panel layout with the given panel object
    centered vertically inside
    """
    return pn.Column(
        pn.VSpacer(),
        panel_object,
        pn.VSpacer(),
    )


def _center_horizontally(panel_object: pn.viewable.Viewable) -> pn.Row:
    """Create a panel layout with the given panel object
    centered horizonally inside
    """
    return pn.Row(
        pn.HSpacer(),
        panel_object,
        pn.HSpacer(),
    )


def with_loading(button_attr_name: str) -> Callable:
    """Decorator to wrap a method with loading/spinner logic when binded to a button"""

    def decorator(func: Callable) -> Callable:
        """Decorator to wrap around any method that encapsulates an event callback"""

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> None | bool:
            button = getattr(self, button_attr_name)
            button.loading = True
            try:
                return func(self, *args, **kwargs)
            finally:
                button.loading = False

        return wrapper

    return decorator
