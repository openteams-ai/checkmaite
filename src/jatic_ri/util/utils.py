"""utils"""

import io
import re
import tempfile
from typing import Any, Union

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


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


# Not expected to be needed after updating to gradient 0.11.0
# See https://gitlab.jatic.net/jatic/morse/gradient/-/issues/643 for more details.
def sanitize_gradient_markdown_text(text: str) -> str:
    """Escape gradient's special characters (e.g. "*" and "_") when rendering Markdown text for slides.

    More info in the gradient documentation: https://jatic.pages.jatic.net/morse/gradient/rai_card_creation/user_guide/reference/markdown_formatting.html

    Args:
        s (str): The input string to escape.

    Returns:
        str: The escaped string.

    Example:
        >>> sanitize_gradient_markdown_text("some_string")
        'some\\_string'
    """
    gradient_special_chars = "*_"
    gradient_escape_char = r"\\"
    return re.sub(f"([{gradient_special_chars}])", rf"{gradient_escape_char}\1", text)
