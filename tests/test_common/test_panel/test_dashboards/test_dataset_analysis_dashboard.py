import json
import os
from pathlib import Path

import jatic_ri
from jatic_ri._common._panel.dashboards.dataset_analysis_dashboard import DatasetAnalysisDashboard



def test_dataset_analysis_dashboard():
    """Test instantiation of DA dashboard and some of the functions.
    Does not run full end to end to avoid heavy compute. Instead, 
    some of the functionality is tested directly.
    """

    app = DatasetAnalysisDashboard(
        task='object_detection',
        output_dir=jatic_ri.DEFAULT_CACHE_ROOT,
    )
    # trigger the visualization to detect errors
    app.panel()

    ## Test loading simple config
    od_config = {
        'task': 'object_detection',
        'shift': {
            'TYPE': 'DatasetShiftTestStage',
        },
    }
    # no config loaded, analysis button should be disabled
    assert app.run_analysis_button.disabled
    # load in the config values
    app.config_file.value = json.dumps(od_config)
    # after loading successful loading of config, analysis button is enabled
    assert not app.run_analysis_button.disabled 

    ## Test dataset loading
    # ensure dataset 2 is not visible to prevent errors
    app.dataset_2_visible = False
    app.dataset_1_selector.value = "Coco dataset"
    app.dataset_1_split_path.value = 'tests/testing_utilities/example_data/coco_dataset'
    app.dataset_1_metadata_path.value = 'tests/testing_utilities/example_data/coco_dataset/ann_file.json'
    load_success = app.load_datasets_from_widgets()
    assert load_success
    assert app.loaded_datasets

    ## test report name generation
    report_title = app._construct_report_filename()
    assert 'Coco' in report_title
