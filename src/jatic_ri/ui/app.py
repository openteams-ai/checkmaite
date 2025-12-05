"""Combined app for JATIC Console, integrating various configurations and dashboards.

To deploy this app, ensure you have the necessary dependencies installed and run:

```bash
poetry run panel serve src/jatic_ri/ui/app.py --show
```
"""

import contextlib

from jatic_ri.ui._panel.dashboards.combined_app import FullApp

# instantiate the app
app = FullApp(local=True)  # pyright: ignore[reportArgumentType]
# create the view for the app
view = app.panel()
# make the view servable
view.servable()


def main() -> None:
    import panel as pn

    t = pn.serve(view, show=True, threaded=True, start=True)

    try:
        t.join()
    except KeyboardInterrupt:
        stop = getattr(t, "stop", None)
        if callable(stop):
            with contextlib.suppress(RuntimeError):
                stop()
