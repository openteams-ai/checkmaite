"""Test XAITKTestStage"""

import pytest
import os

from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages.impls.xaitk_test_stage import (
    XAITKTestStage,
)

ARGS = {
    "name": "XAITKTestStage Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
        },
    },
    "img_batch_size": 1,
}


@pytest.mark.parametrize("use_stage_cache", [True, False])
def test_xaitk_test_stage(use_stage_cache, dummy_model_ic, dummy_dataset_ic, dummy_metric_ic, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_model_ic, model_id="model_1")
    test.load_metric(metric=dummy_metric_ic, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dataset_1")
    test.run(use_stage_cache=use_stage_cache)
    output = test.collect_report_consumables()

    assert len(output) == dummy_dataset_ic.targets.shape[0] * dummy_dataset_ic.targets.shape[1]

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert example_args["layout_arguments"]["text"] == "Model: model\_1\nImage: 0\nGT: dummy\_0\nPred: dummy\_0"
    # assert example_args["layout_arguments"]["image_path"] == f"{os.path.splitext(test.cache_path)[0]}/img_0/class_dummy_0.png"
    example_args["layout_arguments"]["image_path"] == f"{os.path.splitext(test.cache_path)[0]}/img_0/class_dummy_0.png"

    filename = create_deck(output, artifact_dir, 'xaitk')
    assert filename.exists()
