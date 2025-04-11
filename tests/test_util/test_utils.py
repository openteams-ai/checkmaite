from pathlib import Path

from matplotlib.figure import Figure

from jatic_ri.util.utils import create_metrics_bar_plot, save_figure_to_tempfile


def test_create_metrics_bar_plot(metric_results, threshold_od):
    """Test creation of metrics bar plot"""
    fig = create_metrics_bar_plot(metric_results, metric_key="map_50", threshold=threshold_od)
    assert isinstance(fig, Figure)


def test_save_figure_to_tempfile(metric_results, threshold_od):
    """Test saving figure to temporary file"""
    fig = create_metrics_bar_plot(metric_results, metric_key="map_50", threshold=threshold_od)
    filename = save_figure_to_tempfile(fig)
    assert Path(filename).exists
