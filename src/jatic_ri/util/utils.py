"""utils"""

import io
import tempfile
from pathlib import Path
from typing import Any, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def save_figure_to_tempfile(fig: matplotlib.figure.Figure) -> str:
    """Save matplot figure object to temporary file and return filename

    Parameters
    ----------
    fig: matplotlib.figure.Figure
      In-memory mpl figure object

    Returns
    -------
    [str] filename of the temporary file
    """

    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    fp = tempfile.NamedTemporaryFile(delete=False)
    filename = f"{fp.name}.png"

    with open(filename, "wb") as ff:
        ff.write(buf.getvalue())

    buf.close()

    return filename


def temp_image_file(image: PIL.Image.Image, *, suffix: str = ".png") -> Path:  # noqa: D103
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        image.save(f)
        return Path(f.name)


def create_metrics_bar_plot(
    metrics: dict[str, Union[float, Any]],
    metric_key: str,
    threshold: float,
    width: float = 0.75,
) -> matplotlib.figure.Figure:
    """Generate a matplotlib bar chart from metric results

    Parameters
    ----------
    metrics:
        Results of `metric.compute`. Keys are the metric name, values are float
    metric_key:
        Key name for the metric. Will be colored differently than the other bars
    threshold:
        Threshold value, will appear as horizontal line on bar chart
    width:
        [Optional] Width of the bars

    Returns
    -------
    [matplotlib.figure.Figure] Bar chart showing all of the metrics
    """
    default_color = "blue"
    metric_color = "orange"
    threshold_color = "red"

    # create initial canvas
    fig, ax = plt.subplots()

    # set up bar spacing
    index = np.arange(len(metrics))  # the x locations for the bars

    # plot the individual bars
    for idx, (key, value) in enumerate(metrics.items()):
        color = metric_color if key == metric_key else default_color
        ax.bar(index[idx], value, width, color=color)

    # plot the threshold line
    ax.axhline(threshold, color=threshold_color)

    # set title and xtick labels
    ax.set_title("Computed metrics")
    ax.set_xticks(index, metrics.keys())
    if len(metrics) == 1:
        ax.set_xlim(-0.5, 0.5)  # required in order to set width if only one bar
    fig.tight_layout()

    return fig
