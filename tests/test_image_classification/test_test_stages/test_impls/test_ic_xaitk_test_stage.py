"""Test XAITKTestStage"""

import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.image_classification.test_stages import XAITKTestStage

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


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_test_stage_rise(fake_ic_dataset_default, fake_ic_model_default, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(RISE_ARGS)
    # load the maite compliant model
    run = test.run(use_stage_cache=False, models=[fake_ic_model_default], datasets=[fake_ic_dataset_default])
    output = run.collect_report_consumables(threshold=0.5)

    assert len(output) == len(fake_ic_dataset_default) * fake_ic_dataset_default[0][1].shape[0]

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert example_args["layout_arguments"]["text"] == "Model: model\\_1\nImage: 0\nGT: date\nPred: apple"
    assert example_args["layout_arguments"]["image_path"].is_file()

    filename = create_deck(output, artifact_dir, "xaitk")
    assert filename.exists()


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_test_stage_mc_rise(fake_ic_dataset_default, fake_ic_model_default, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(MC_RISE_ARGS)
    # load the maite compliant model
    run = test.run(use_stage_cache=False, models=[fake_ic_model_default], datasets=[fake_ic_dataset_default])
    output = run.collect_report_consumables(threshold=0.5)

    assert (
        len(output) == len(fake_ic_dataset_default) * fake_ic_dataset_default[0][1].shape[0] * 2
    )  # multiply by number of fill colors

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "OneImageText"
    assert example_args["layout_arguments"]["title"] == "**XAITK Saliency Map**: 0 \n"
    assert (
        example_args["layout_arguments"]["text"]
        == "Model: model\\_1\nImage: 0\nFill Color: [255, 0, 0]\nGT: date\nPred: apple"
    )
    assert example_args["layout_arguments"]["image_path"].is_file()
    filename = create_deck(output, artifact_dir, "xaitk")
    assert filename.exists()
