"""Test XAITKTestStage"""

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


def test_xaitk_test_stage(dummy_xaitk_model, dummy_xaitk_dataset, dummy_metric) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_xaitk_model, model_id="model_1")
    test.load_metric(metric=dummy_metric, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_xaitk_dataset, dataset_id="dataset_1")
    test.run()
    test.collect_report_consumables()


def test_xaitk_test_stage_no_cache(dummy_xaitk_model, dummy_xaitk_dataset, dummy_metric) -> None:
    """Test XAITKTestStage implementation without caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_xaitk_model, model_id="model_1")
    test.load_metric(metric=dummy_metric, metric_id="metric_1")
    test.load_threshold(threshold=10)
    test.load_dataset(dataset=dummy_xaitk_dataset, dataset_id="dataset_1")
    test.run(use_cache=False)
    test.collect_report_consumables()
