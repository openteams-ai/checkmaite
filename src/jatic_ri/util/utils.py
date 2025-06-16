"""utils"""

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from IPython.display import HTML


def save_figure_to_tempfile(fig: matplotlib.figure.Figure) -> str:
    """Save matplotlib figure object to a temporary file and return its filename.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        In-memory mpl figure object.

    Returns
    -------
    str
        Filename of the temporary file.
    """

    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    fp = tempfile.NamedTemporaryFile(delete=False)
    filename = f"{fp.name}.png"

    with open(filename, "wb") as ff:
        ff.write(buf.getvalue())

    buf.close()

    return filename


def temp_image_file(image: PIL.Image.Image, *, suffix: str = ".png") -> Path:
    """Save a PIL Image to a temporary file and return its path.

    Parameters
    ----------
    image : PIL.Image.Image
        The image to save.
    suffix : str, optional
        The suffix for the temporary file (default is ".png").

    Returns
    -------
    pathlib.Path
        The path to the temporary image file.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        image.save(f)
        return Path(f.name)


def create_metrics_bar_plot(
    metrics: dict[str, float | Any],
    metric_key: str,
    threshold: float,
    width: float = 0.75,
) -> matplotlib.figure.Figure:
    """Generate a matplotlib bar chart from metric results.

    Parameters
    ----------
    metrics : dict[str, float | Any]
        Results of `metric.compute`. Keys are the metric name, values are float.
    metric_key : str
        Key name for the metric. Will be colored differently than the other bars.
    threshold : float
        Threshold value, will appear as horizontal line on bar chart.
    width : float, optional
        Width of the bars (default is 0.75).

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart showing all of the metrics.
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
    plt.close(fig)

    return fig


def create_expandable_output(outputs: dict | list, max_preview_length: int = 100) -> HTML:
    """Display test stage results concisely with collapsible sections for long values.

    Parameters
    ----------
    outputs : dict | list
        The data to display. If a dictionary, its items (key-value pairs) are
        displayed. If a list, it is treated as a single entry under the name "Output".
    max_preview_length : int, optional
        The maximum number of characters to show in the preview before
        collapsing the content (default is 100).

    Returns
    -------
    IPython.display.HTML
        An HTML object representing the expandable output.
    """
    items = outputs.items() if isinstance(outputs, dict) else [("Output", outputs)]
    parts = []
    for name, value in items:
        text = str(value)
        if len(text) <= max_preview_length:
            parts.append(f"""
            <div style="margin-bottom:15px;">
                <strong>{name}:</strong> {text}
            </div>""")
        else:
            preview = text[:max_preview_length] + "..."
            parts.append(f"""
            <div style="margin-bottom:15px;">
                <strong>{name}:</strong> {preview}
                <details style="margin-top:5px;">
                    <summary>Show full output</summary>
                    <pre style="background-color:#f5f5f5;padding:10px;margin-top:5px;">{text}</pre>
                </details>
            </div>
            """)
    return HTML("".join(parts))


def id_hash(**kwargs: Any) -> str:
    """Generate a consistent hash from keyword arguments.

    Parameters
    ----------
    **kwargs : Any
        Key-value pairs to include in the hash generation

    Returns
    -------
    str
        First 8 characters of the SHA-256 hash of the JSON-serialized kwargs
    """
    return hashlib.sha256(json.dumps(kwargs, default=str, sort_keys=True).encode()).hexdigest()[:8]
