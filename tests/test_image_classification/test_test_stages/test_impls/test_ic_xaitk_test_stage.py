"""Test XAITKTestStage"""

import os

from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages.impls.xaitk_test_stage import (
    XAITKTestStage,
)

RISE_ARGS = {
    "name": "XAITKTestStage RISE Example",
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

MC_RISE_ARGS = {
    "name": "XAITKTestStage MC-RISE Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
            "fill_colors": [[255, 0, 0], [0, 255, 0]],
        },
    },
    "img_batch_size": 1,
}


def test_xaitk_test_stage_rise(dummy_model_ic, dummy_dataset_ic, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(RISE_ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_model_ic, model_id="model_1")
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dataset_1")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()

    assert len(output) == len(dummy_dataset_ic) * dummy_dataset_ic[0][1].shape[0]

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert example_args["layout_arguments"]["text"] == "Model: model\\_1\nImage: 0\nGT: dummy\\_0\nPred: dummy\\_0"
    assert (
        str(example_args["layout_arguments"]["image_path"])
        == f"{os.path.splitext(test.cache_path)[0]}/img_0/class_dummy_0.png"
    )

    filename = create_deck(output, artifact_dir, "xaitk")
    assert filename.exists()


def test_xaitk_test_stage_mc_rise(dummy_model_ic, dummy_dataset_ic, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(MC_RISE_ARGS)
    # load the maite compliant model
    test.load_model(model=dummy_model_ic, model_id="model_1")
    test.load_dataset(dataset=dummy_dataset_ic, dataset_id="dataset_1")
    test.run(use_stage_cache=False)
    output = test.collect_report_consumables()

    assert (
        len(output) == len(dummy_dataset_ic) * dummy_dataset_ic[0][1].shape[0] * 2
    )  # multiply by number of fill colors

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert (
        example_args["layout_arguments"]["text"]
        == "Model: model\\_1\nImage: 0\nFill Color: [255, 0, 0]\nGT: dummy\\_0\nPred: dummy\\_0"
    )
    assert (
        str(example_args["layout_arguments"]["image_path"])
        == f"{os.path.splitext(test.cache_path)[0]}/img_0/color_[255, 0, 0]_class_dummy_0.png"
    )

    filename = create_deck(output, artifact_dir, "xaitk")
    assert filename.exists()
