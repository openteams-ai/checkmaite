import os
from pathlib import Path

import pytest
from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
from nrtk.impls.perturb_image_factory import PerturberOneStepFactory

##############################
# UI Configuration #
##############################


@pytest.fixture(scope="session")
def reallabel_config_od():
    """Default output configuration settings from the reallabel
    panel app"""
    pytest.importorskip("checkmaite_plugins")
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.object_detection.reallabel_app import RealLabelApp as RealLabelAppOD

    reallabel_app = RealLabelAppOD()
    reallabel_app._run_export()
    return reallabel_app.output_test_stages["reallabel_test_stage"]


@pytest.fixture(scope="session")
def survivor_config_od():
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    pytest.importorskip("checkmaite_plugins")
    from checkmaite.ui.configuration_pages.object_detection.survivor_app import SurvivorAppOD

    survivor_app = SurvivorAppOD()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]


@pytest.fixture(scope="session")
def survivor_config_ic():
    pytest.importorskip("checkmaite_plugins")
    """Default output configuration settings from the survivor
    panel app"""
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.image_classification.survivor_app import SurvivorAppIC

    survivor_app = SurvivorAppIC()
    survivor_app._run_export()
    return survivor_app.output_test_stages["survivor_test_stage"]


@pytest.fixture(scope="session")
def nrtk_config_od():
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.object_detection.nrtk_app import NRTKAppOD

    nrtk_app = NRTKAppOD()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = PerturberOneStepFactory
    nrtk_app.theta_key.value = "factor"
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    return nrtk_app.output_test_stages["NRTKAppOD_0"]


@pytest.fixture(scope="session")
def nrtk_config_ic():
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.image_classification.nrtk_app import NRTKAppIC

    nrtk_app = NRTKAppIC()
    nrtk_app.panel()
    nrtk_app.perturber_select.value = BrightnessPerturber
    nrtk_app.factory_selector.value = PerturberOneStepFactory
    nrtk_app.theta_key.value = "factor"
    nrtk_app.theta_value.value = 10.0
    nrtk_app.name_input.value = "TestFactory"
    nrtk_app.add_test_stage_callback(None)
    nrtk_app._run_export()
    return nrtk_app.output_test_stages["NRTKAppIC_0"]


@pytest.fixture(scope="session")
def xaitk_config_od():
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.object_detection.xaitk_app import XAITKAppOD

    xaitk_app = XAITKAppOD()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKAppOD_0"]


@pytest.fixture(scope="session")
def xaitk_config_ic():
    # import inside the fixture to improve test startup time
    from checkmaite.ui.configuration_pages.image_classification.xaitk_app import XAITKAppIC

    xaitk_app = XAITKAppIC()
    xaitk_app._run_export()
    return xaitk_app.output_test_stages["XAITKAppIC_0"]


@pytest.fixture(scope="session")
def feasibility_config_od():
    return {
        "TYPE": "DatasetFeasibilityTestStage",
    }


@pytest.fixture(scope="session")
def feasibility_config_ic(feasibility_config_od):
    return feasibility_config_od


@pytest.fixture(scope="session")
def bias_config_od():
    return {
        "TYPE": "DatasetBiasTestStage",
    }


@pytest.fixture(scope="session")
def bias_config_ic(bias_config_od):
    return bias_config_od


@pytest.fixture(scope="session")
def cleaning_config_od():
    return {
        "TYPE": "DatasetCleaningTestStage",
    }


@pytest.fixture(scope="session")
def cleaning_config_ic(cleaning_config_od):
    return cleaning_config_od


@pytest.fixture(scope="session")
def shift_config_od():
    return {
        "TYPE": "DatasetShiftTestStage",
    }


@pytest.fixture(scope="session")
def shift_config_ic(shift_config_od):
    return shift_config_od


@pytest.fixture(scope="session")
def baseline_eval_config_od():
    # baseline evaluation (maite) requires no data inside the config
    return {
        "TYPE": "BaselineEvaluationTestStage",
    }


@pytest.fixture(scope="session")
def baseline_eval_config_ic(baseline_eval_config_od):
    # baseline evaluation (maite) requires no data inside the config
    return baseline_eval_config_od


@pytest.fixture
def artifact_dir(tmp_path):
    """
    Provides a directory for storing test artifacts.

    By default, artifacts go to a temporary directory that is removed
    after the tests. Set SAVE_POETRY_ARTIFACTS=true to persist
    artifacts to ARTIFACT_DIR (or fallback to tmp if unset).
    """
    _save = os.environ.get("SAVE_POETRY_ARTIFACTS", "false").strip().lower() == "true"
    _env_dir = os.environ.get("ARTIFACT_DIR")

    if _save and _env_dir:
        path = Path(_env_dir).expanduser().resolve()
    else:
        path = tmp_path

    path.mkdir(parents=True, exist_ok=True)
    return path
