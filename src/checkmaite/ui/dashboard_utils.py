from collections.abc import Callable
from typing import Any

import panel as pn


# ruff: noqa: I001
from checkmaite.core.image_classification import (
    DataevalBias as DataevalBiasIC,
    DataevalBiasConfig as DataevalBiasConfigIC,
    DataevalCleaning as DataevalCleaningIC,
    DataevalCleaningConfig as DataevalCleaningConfigIC,
    DataevalFeasibility as DataevalFeasibilityIC,
    DataevalFeasibilityConfig as DataevalFeasibilityConfigIC,
    DataevalShift as DataevalShiftIC,
    DataevalShiftConfig as DataevalShiftConfigIC,
    MaiteEvaluation as MaiteEvaluationIC,
    MaiteEvaluationConfig as MaiteEvaluationConfigIC,
    NrtkRobustness as NrtkRobustnessIC,
    NrtkRobustnessConfig as NrtkRobustnessConfigIC,
    XaitkExplainable as XaitkExplainableIC,
    XaitkExplainableConfig as XaitkExplainableConfigIC,
    #    SurvivorTestStage as SurvivorTestStageIC,
)

# ruff: noqa: I001
from checkmaite.core.object_detection import (
    DataevalBias as DataevalBiasOD,
    DataevalBiasConfig as DataevalBiasConfigOD,
    DataevalCleaning as DataevalCleaningOD,
    DataevalCleaningConfig as DataevalCleaningConfigOD,
    DataevalFeasibility as DataevalFeasibilityOD,
    DataevalFeasibilityConfig as DataevalFeasibilityConfigOD,
    DataevalShift as DataevalShiftOD,
    DataevalShiftConfig as DataevalShiftConfigOD,
    MaiteEvaluation as MaiteEvaluationOD,
    MaiteEvaluationConfig as MaiteEvaluationConfigOD,
    NrtkRobustness as NrtkRobustnessOD,
    NrtkRobustnessConfig as NrtkRobustnessConfigOD,
    XaitkExplainable as XaitkExplainableOD,
    XaitkExplainableConfig as XaitkExplainableConfigOD,
    #    RealLabelTestStage as RealLabelTestStageOD,
    #    SurvivorTestStage as SurvivorTestStageOD,
)


def get_capability_from_app_config_od(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Initialize test stage object based on config dictionary"""

    # Some capabilities take no configuration parameters, so CONFIG key will not exist.
    capability_config = config.get("CONFIG", {})

    # if config["TYPE"] == "RealLabelTestStage":
    #     reallabel_config = RealLabelConfig(**capability_config)
    #     return RealLabelTestStageOD(config=reallabel_config)
    if config["TYPE"] == "NRTKTestStage":
        return {
            "stage": NrtkRobustnessOD(),
            "config": NrtkRobustnessConfigOD(**capability_config),
        }
    if config["TYPE"] == "XAITKTestStage":
        return {
            "stage": XaitkExplainableOD(),
            "config": XaitkExplainableConfigOD(**capability_config),
        }
    # if config["TYPE"] == "SurvivorTestStage":
    #     return SurvivorTestStageOD(**capability_config)
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return {
            "stage": MaiteEvaluationOD(),
            "config": MaiteEvaluationConfigOD(**capability_config),
        }
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return {
            "stage": DataevalFeasibilityOD(),
            "config": DataevalFeasibilityConfigOD(**capability_config),
        }
    if config["TYPE"] == "DatasetBiasTestStage":
        return {
            "stage": DataevalBiasOD(),
            "config": DataevalBiasConfigOD(**capability_config),
        }
    if config["TYPE"] == "DatasetCleaningTestStage":
        return {"stage": DataevalCleaningOD(), "config": DataevalCleaningConfigOD(**capability_config)}
    if config["TYPE"] == "DatasetShiftTestStage":
        return {
            "stage": DataevalShiftOD(),
            "config": DataevalShiftConfigOD(**capability_config),
        }

    raise RuntimeError(f'Unable to instantiate TestStage object from config: {config["TYPE"]}')


def get_capability_from_app_config_ic(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Initialize test stage object based on config dictionary"""

    # Some capabilities take no configuration parameters, so CONFIG key will not exist.
    capability_config = config.get("CONFIG", {})

    if config["TYPE"] == "NRTKTestStage":
        return {
            "stage": NrtkRobustnessIC(),
            "config": NrtkRobustnessConfigIC(**capability_config),
        }
    if config["TYPE"] == "XAITKTestStage":
        return {
            "stage": XaitkExplainableIC(),
            "config": XaitkExplainableConfigIC(**capability_config),
        }
    # if config["TYPE"] == "SurvivorTestStage":
    #     return SurvivorTestStageIC(**capability_config)
    if config["TYPE"] == "HeartTestStage":
        raise RuntimeError("Heart test stage is not currently supported.")
    if config["TYPE"] == "BaselineEvaluationTestStage":
        return {
            "stage": MaiteEvaluationIC(),
            "config": MaiteEvaluationConfigIC(**capability_config),
        }
    if config["TYPE"] == "DatasetFeasibilityTestStage":
        return {
            "stage": DataevalFeasibilityIC(),
            "config": DataevalFeasibilityConfigIC(**capability_config),
        }
    if config["TYPE"] == "DatasetBiasTestStage":
        return {
            "stage": DataevalBiasIC(),
            "config": DataevalBiasConfigIC(**capability_config),
        }
    if config["TYPE"] == "DatasetCleaningTestStage":
        return {"stage": DataevalCleaningIC(), "config": DataevalCleaningConfigIC(**capability_config)}
    if config["TYPE"] == "DatasetShiftTestStage":
        return {
            "stage": DataevalShiftIC(),
            "config": DataevalShiftConfigIC(**capability_config),
        }
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
