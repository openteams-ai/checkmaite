"""Test XAITKTestStage"""

import os
from pathlib import Path

import pytest
from gradient.templates_and_layouts.create_deck import create_deck
from torch import as_tensor, equal

from jatic_ri.object_detection.test_stages.impls.xaitk_test_stage import XAITKTestStage

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
    "img_batch_size": 1,
}


@pytest.mark.parametrize("use_stage_cache", [True, False])
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_xaitk_test_stage(use_stage_cache, fake_od_model_default, fake_od_dataset_default, artifact_dir) -> None:
    """Test XAITKTestStage implementation with caching"""

    test = XAITKTestStage(ARGS)
    # load the maite compliant model
    test.load_model(model=fake_od_model_default, model_id="model_1")
    test.load_dataset(dataset=fake_od_dataset_default, dataset_id="dataset_1")
    test.run(use_stage_cache=use_stage_cache)
    output = test.collect_report_consumables()

    assert len(output) == len(fake_od_dataset_default) * len(fake_od_dataset_default[0][1].scores)

    example_args = output[0]

    assert all(required_key in example_args for required_key in ("deck", "layout_name", "layout_arguments"))

    assert example_args["layout_name"] == "TwoItem"
    assert example_args["layout_arguments"]["title"] == (
        "XAITK Saliency Map -- " f"Image ID: {fake_od_dataset_default[0][2]['id']}, " "Detection: 0"
    )
    assert example_args["layout_arguments"]["left_item"] == Path(
        f"{os.path.splitext(test.cache_path)[0]}/img_0/det_0.png"
    )
    assert example_args["layout_arguments"]["right_item"] == (
        "**Model:** model_1\n"
        "**Image ID**: some_string\n"
        "**Prediction:** date\n"
        "**Confidence:** 0.90\n\n\n"
        "Note: The Confidence is the metric score that the given detection had in the original "
        "(un-occluded) image.  Pixel relevance is normalized on scale from 0 to 1."
    )

    filename = create_deck(output, artifact_dir, "xaitk")
    assert filename.exists()


def test_xaitk_temp_dataset(fake_od_dataset_default, fake_od_model_default) -> None:
    """Test XAITKDetectionBaselineDataset item retrieval"""
    # Initialize a test stage and extract the temporary dataset.  Limit 2 detections.
    temp_dataset = XAITKTestStage(ARGS).XAITKDetectionBaselineDataset(
        fake_od_dataset_default, fake_od_model_default, dets_limit=2
    )
    # Validate the length of the dataset
    assert len(temp_dataset) == len(fake_od_dataset_default)
    # Retrieve an item and validate its structure and content
    for i in range(len(temp_dataset)):
        # Confirm images are the same
        assert equal(temp_dataset[i][0], fake_od_dataset_default[i][0])

        # Get the model's detetcions on original dataset and the highest scoring detection
        dets_i = fake_od_model_default(fake_od_dataset_default[i])[0]
        max_score_i = dets_i.scores.argmax()

        # Validate the first detection in the temp dataset is the highest scoring from the original
        assert equal(as_tensor(temp_dataset[i][1].boxes)[0], dets_i.boxes[max_score_i])
        assert equal(as_tensor(temp_dataset[i][1].labels)[0], dets_i.labels[max_score_i])
        assert equal(as_tensor(temp_dataset[i][1].scores)[0], dets_i.scores[max_score_i])

        # Validate that there are no more than 2 detections per image
        assert len(as_tensor(temp_dataset[i][1].boxes)) <= 2
