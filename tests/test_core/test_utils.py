from pathlib import Path

import pytest
import torch

from jatic_ri.core._utils import set_device


@pytest.fixture
def threshold_od() -> float:
    return 0.3


@pytest.fixture
def metric_results() -> dict:
    return {
        "map50": 0.12,
        "airports": 0.43,
        "elephants": 0.56,
    }


def test_set_device_default_behavior():
    if torch.cuda.is_available():
        expected_device = "cuda"
    elif torch.backends.mps.is_available():
        expected_device = "mps"
    else:
        expected_device = "cpu"
    device = set_device(None)
    assert device.type == expected_device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_set_device_cuda():
    assert set_device(None).type == "cuda"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_set_device_mps():
    assert set_device(None).type == "mps"


def test_set_device_idempotency():
    device1 = set_device("cpu")
    device2 = set_device(device1)
    assert device1 == device2


def test_set_device_with_torch_device():
    for dev in ["cpu", "cuda", "mps"]:
        expected_device = torch.device(dev)
        actual_device = set_device(expected_device)
        assert actual_device == expected_device


def test_create_metrics_bar_plot(metric_results, threshold_od):
    from matplotlib.figure import Figure

    from jatic_ri.core.report._plotting_utils import create_metrics_bar_plot

    fig = create_metrics_bar_plot(metric_results, metric_key="map_50", threshold=threshold_od)
    assert isinstance(fig, Figure)


def test_save_figure_to_tempfile(metric_results, threshold_od):
    from jatic_ri.core.report._plotting_utils import create_metrics_bar_plot, save_figure_to_tempfile

    fig = create_metrics_bar_plot(metric_results, metric_key="map_50", threshold=threshold_od)
    filename = save_figure_to_tempfile(fig)
    assert Path(filename).exists
