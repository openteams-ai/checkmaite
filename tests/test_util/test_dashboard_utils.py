import pytest

from jatic_ri.util.dashboard_utils import create_download_link, rehydrate_test_stage_od, rehydrate_test_stage_ic


def test_create_download_link():
    path = 'testfile.txt'
    link_str = create_download_link(path=path)

    assert path in link_str


@pytest.mark.parametrize(
    "config_fixture_name",
    ['reallabel_config_od', 'nrtk_config_od', 'survivor_config_od', 'xaitk_config_od', 'feasibility_config_od', 'bias_config_od', 'linting_config_od', 'shift_config_od', 'baseline_eval_config_od'],
)
def test_rehydrate_test_stage_od(config_fixture_name, request):
    """Ensure the test stage can be constructed from config values"""
    config = request.getfixturevalue(config_fixture_name)
    
    rehydrate_test_stage_od(config=config)


@pytest.mark.parametrize(
    "config_fixture_name",
    ['survivor_config_ic', 'nrtk_config_ic', 'xaitk_config_ic', 'feasibility_config_ic', 'bias_config_ic', 'linting_config_ic', 'shift_config_ic', 'baseline_eval_config_ic'],
)
def test_rehydrate_test_stage_ic(config_fixture_name, request):
    """Ensure the test stage can be constructed from config values"""
    config = request.getfixturevalue(config_fixture_name)
    
    rehydrate_test_stage_ic(config=config)


def test_rehydrate_test_stage_od_unrecognized():
    heart_config = {
        'TYPE': 'HeartTestStage',
    }
    with pytest.raises(RuntimeError, match=r"\bHeart test stage\b"):
        rehydrate_test_stage_od(config=heart_config)

    unknown_config = {
        'TYPE': 'unknown_stage',
    }
    with pytest.raises(RuntimeError, match=r"\bUnable to instantiate\b"):
        rehydrate_test_stage_od(config=unknown_config)


def test_rehydrate_test_stage_ic_unrecognized():
    heart_config = {
        'TYPE': 'HeartTestStage',
    }
    with pytest.raises(RuntimeError, match=r"\bHeart test stage\b"):
        rehydrate_test_stage_ic(config=heart_config)

    unknown_config = {
        'TYPE': 'unknown_stage',
    }
    with pytest.raises(RuntimeError, match=r"\bUnable to instantiate\b"):
        rehydrate_test_stage_ic(config=unknown_config)
