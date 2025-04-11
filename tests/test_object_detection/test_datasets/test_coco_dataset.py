import torch

from jatic_ri.object_detection.datasets import CocoDetectionDataset, DetectionTarget


def test_coco_dataset():
    coco_dataset = CocoDetectionDataset(
        root="tests/testing_utilities/example_data/coco_dataset",
        ann_file="tests/testing_utilities/example_data/coco_dataset/ann_file.json",
    )
    assert len(coco_dataset) == 4
    element = coco_dataset[0]
    assert isinstance(element[0], torch.Tensor)
    assert element[0].ndim == 3
    assert isinstance(element[1], DetectionTarget)
    assert element[1].boxes.ndim == 2
    assert element[1].labels.ndim == 1
    assert element[1].scores.ndim == 1
    assert element[1].scores.shape == (14,)
    assert isinstance(element[2], dict)
    assert coco_dataset.metadata["index2label"][1] == "person"
    assert coco_dataset.metadata["index2label"][2] == "bicycle"
