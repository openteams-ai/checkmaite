"""Test XAITKTestStage"""

import os
from pathlib import Path
import pytest
import os

from jatic_ri.object_detection.test_stages.impls.xaitk_test_stage import (
    XAITKTestStage,
)
from gradient.templates_and_layouts.create_deck import create_deck

ARGS = {
    "name": "XAITKTestStage Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
        "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
        },
    },
}


@pytest.mark.parametrize("use_cache", [True, False])
def test_xaitk_test_stage(use_cache, dummy_xaitk_model, dummy_xaitk_dataset, dummy_metric_od, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_xaitk_model, model_id="model_1")
    test.load_metric(metric=dummy_metric_od, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_xaitk_dataset, dataset_id="dataset_1")
    test.run(use_cache=use_cache)
    output = test.collect_report_consumables()

    assert len(output) == len(dummy_xaitk_dataset) * len(dummy_xaitk_dataset[0][1].scores)

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert example_args["layout_arguments"]["text"] == "Model: model\_1\nImage: 0\nGT: dummy\_0\nPred: dummy\_0"
    assert example_args["layout_arguments"]["image_path"] == Path(f"{os.path.splitext(test.cache_path)[0]}/img_0/det_0.png")
    
    filename = create_deck(output, artifact_dir, 'xaitk')
    assert filename.exists()
