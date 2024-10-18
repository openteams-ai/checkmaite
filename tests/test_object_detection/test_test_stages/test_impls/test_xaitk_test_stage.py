"""Test XAITKTestStage"""

import pytest

from jatic_ri.object_detection.test_stages.impls.xaitk_test_stage import (
    XAITKTestStage,
)

ARGS = {
    "name": "XAITKTestStage Example",
    "id2label": {0: "dummy_0", 1: "dummy_1", 2: "dummy_2", 3: "dummy_3", 4: "dummy_4"},
    "GenerateObjectDetectorBlackboxSaliency": {
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
def test_xaitk_test_stage(use_cache, dummy_xaitk_model, dummy_xaitk_dataset, dummy_metric_od) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_xaitk_model, model_id="model_1")
    test.load_metric(metric=dummy_metric_od, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_xaitk_dataset, dataset_id="dataset_1")
    output = test.collect_report_consumables()
    assert len(output) == 0
    test.run(use_cache=use_cache)
    output = test.collect_report_consumables()

    assert len(output) == len(dummy_xaitk_dataset) * len(dummy_xaitk_dataset[0][1].scores)

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert example_args["layout_arguments"]["text"] == "Model: model_1\nImage: 0\nGT: dummy_0\nPred: dummy_0"
    assert example_args["layout_arguments"]["image_path"] == f"{test.cache_path[:-5]}/img_0/det_0.png"
