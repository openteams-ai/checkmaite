from IPython.display import HTML


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
            parts.append(
                f"""
            <div style="margin-bottom:15px;">
                <strong>{name}:</strong> {text}
            </div>"""
            )
        else:
            preview = text[:max_preview_length] + "..."
            parts.append(
                f"""
            <div style="margin-bottom:15px;">
                <strong>{name}:</strong> {preview}
                <details style="margin-top:5px;">
                    <summary>Show full output</summary>
                    <pre style="background-color:#f5f5f5;padding:10px;margin-top:5px;">{text}</pre>
                </details>
            </div>
            """
            )
    return HTML("".join(parts))
