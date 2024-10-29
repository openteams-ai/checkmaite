from jatic_ri.util.utils import save_figure_to_tempfile, create_metrics_bar_plot
from matplotlib.figure import Figure
from pathlib import Path


def test_create_metrics_bar_plot(metric_results, threshold_od):
    """test creation of metrics bar plot"""
    fig = create_metrics_bar_plot(metric_results, metric_key='map_50', threshold=threshold_od)
    assert isinstance(fig, Figure)


def test_save_figure_to_tempfile(metric_results, threshold_od):
    """test saving figure to temporary file"""
    fig = create_metrics_bar_plot(metric_results, metric_key='map_50', threshold=threshold_od)
    filename = save_figure_to_tempfile(fig)
    assert Path(filename).exists
