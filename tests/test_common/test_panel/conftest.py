import copy

import pytest


@pytest.fixture(scope="session")
def json_config_da_od(reallabel_config_od, survivor_config_od, bias_config_od, cleaning_config_od, shift_config_od):
    """Json configuration output from the dataset analysis object detection app"""
    return {
        "task": "object_detection",
        "reallabel": reallabel_config_od,
        "survivor": survivor_config_od,
        "bias": bias_config_od,
        "cleaning": cleaning_config_od,
        # 'shift': shift_config_od,  # uncomment after shift bug fix
    }


@pytest.fixture(scope="session")
def json_config_da_ic(survivor_config_ic, bias_config_ic, cleaning_config_ic, feasibility_config_ic, shift_config_ic):
    """Json configuration output from the dataset analysis image classification app"""
    return {
        "task": "image_classification",
        "survivor": survivor_config_ic,
        "bias": bias_config_ic,
        "cleaning": cleaning_config_ic,
        # 'shift': shift_config_ic,  # uncomment after shift bug fix
    }


@pytest.fixture(scope="session")
def json_config_me_od(nrtk_config_od, xaitk_config_od, baseline_eval_config_od):
    """Json configuration output from the model evaluation object detection app"""
    return {
        "task": "object_detection",
        "nrtk1": nrtk_config_od,
        "nrtk2": copy.deepcopy(nrtk_config_od),
        "xaitk": xaitk_config_od,
        "baseline_eval": baseline_eval_config_od,
    }


@pytest.fixture(scope="session")
def json_config_me_ic(nrtk_config_ic, xaitk_config_ic, baseline_eval_config_ic):
    """Json configuration output from the model evaluation image classification app"""
    return {
        "task": "image_classification",
        "nrtk": nrtk_config_ic,
        # 'xaitk': xaitk_config_ic,
        "baseline_eval": baseline_eval_config_ic,
    }
