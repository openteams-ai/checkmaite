"""Combined app for JATIC Console, integrating various configurations and dashboards.

To deploy this app, ensure you have the necessary dependencies installed and run:

```bash
poetry run panel serve src/jatic_ri/deploy/app.py --show
```
"""

from jatic_ri._common._panel.configurations.combined_app import FullApp

# instantiate the app
app = FullApp(local=True)  # pyright: ignore[reportArgumentType]
# create the view for the app
view = app.panel()
# make the view servable
view.servable()
