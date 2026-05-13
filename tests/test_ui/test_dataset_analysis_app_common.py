from checkmaite.ui._common import dataset_analysis_app as da_module
from checkmaite.ui._common.dataset_analysis_app import DAConfigurationLandingPage


def test_da_landing_page_preserves_routing_output_and_view_invariants(monkeypatch) -> None:
    monkeypatch.setattr(da_module, "HAS_SURVIVOR", True)
    monkeypatch.setattr(da_module, "HAS_REALLABEL", True)

    app = DAConfigurationLandingPage(task="object_detection")
    assert app.next_parameter == "DatasetAnalysisDashboard"

    app.show_survivor_config = True
    app._update_next_parameter()
    assert app.next_parameter == "Configure SurvivorOD"

    app.task = "image_classification"
    app._update_next_parameter()
    assert app.next_parameter == "Configure SurvivorIC"

    app.show_reallabel_config = True
    app._update_next_parameter()
    assert app.next_parameter == "Configure Reallabel"

    for checkbox in (app.bias, app.feasibility, app.shift, app.cleaning):
        checkbox.value = True
    next_stage, task, output, local = app.output()
    assert (next_stage, task, local) == ("Configure SurvivorIC", "image_classification", app.local)
    assert set(output) == {"task", "bias", "feasibility", "shift", "cleaning"}

    od_app = DAConfigurationLandingPage(task="object_detection")
    od_app.view_analysis_tools()
    od_app.show_survivor_config = True
    assert od_app.output()[0] == "Configure SurvivorOD"
