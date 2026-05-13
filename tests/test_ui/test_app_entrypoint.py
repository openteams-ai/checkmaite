import importlib


def test_ui_app_entrypoint_instantiates_app_smoke() -> None:
    # Importing the entrypoint instantiates FullApp, builds its Panel view, and
    # marks the view servable; this smoke test catches import-time app wiring failures.
    importlib.import_module("checkmaite.ui.app")
