import pytest


def test_create_download_link():
    from jatic_ri.ui.dashboard_utils import create_download_link

    path = "testfile.txt"
    link_str = create_download_link(path=path)

    assert path in link_str


@pytest.mark.parametrize(
    "config_fixture_name",
    [
        # "reallabel_config_od",
        "nrtk_config_od",
        # "survivor_config_od",
        "xaitk_config_od",
        "feasibility_config_od",
        "bias_config_od",
        "cleaning_config_od",
        "shift_config_od",
        "baseline_eval_config_od",
    ],
)
def test_get_capability_from_app_config_od(config_fixture_name, request):
    from jatic_ri.ui.dashboard_utils import get_capability_from_app_config_od

    config = request.getfixturevalue(config_fixture_name)

    get_capability_from_app_config_od(config=config)


@pytest.mark.parametrize(
    "config_fixture_name",
    [
        # "survivor_config_ic",
        "nrtk_config_ic",
        "xaitk_config_ic",
        "feasibility_config_ic",
        "bias_config_ic",
        "cleaning_config_ic",
        "shift_config_ic",
        "baseline_eval_config_ic",
    ],
)
def test_get_capability_from_app_config_ic(config_fixture_name, request):
    from jatic_ri.ui.dashboard_utils import get_capability_from_app_config_ic

    config = request.getfixturevalue(config_fixture_name)

    get_capability_from_app_config_ic(config=config)


def test_get_capability_from_app_config_od_unrecognized():
    from jatic_ri.ui.dashboard_utils import get_capability_from_app_config_od

    heart_config = {
        "TYPE": "HeartTestStage",
    }
    with pytest.raises(RuntimeError, match=r"\bHeart test stage\b"):
        get_capability_from_app_config_od(config=heart_config)

    unknown_config = {
        "TYPE": "unknown_stage",
    }
    with pytest.raises(RuntimeError, match=r"\bUnable to instantiate\b"):
        get_capability_from_app_config_od(config=unknown_config)


def test_get_capability_from_app_config_ic_unrecognized():
    from jatic_ri.ui.dashboard_utils import get_capability_from_app_config_ic

    heart_config = {
        "TYPE": "HeartTestStage",
    }
    with pytest.raises(RuntimeError, match=r"\bHeart test stage\b"):
        get_capability_from_app_config_ic(config=heart_config)

    unknown_config = {
        "TYPE": "unknown_stage",
    }
    with pytest.raises(RuntimeError, match=r"\bUnable to instantiate\b"):
        get_capability_from_app_config_ic(config=unknown_config)
