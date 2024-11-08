from jatic_ri.object_detection.datasets import YoloDetectionDataset, DetectionTarget
import torch

import numpy as np

def test_yolo_dataset():
    yolo_dataset = YoloDetectionDataset(
        yaml_dataset='tests/test_object_detection/data/yolo_dataset/dataset.yaml',
        ann_dir='tests/test_object_detection/data/yolo_dataset/ann_dir',
    )
    assert len(yolo_dataset) == 4
    element = yolo_dataset[0]
    assert isinstance(element[0], torch.Tensor) 
    assert element[0].ndim == 3
    assert isinstance(element[1], DetectionTarget)
    assert element[1].boxes.ndim == 2
    assert element[1].labels.ndim == 1
    assert element[1].scores.ndim == 1
    assert element[1].scores.shape == (14,)
    assert element[2] == {}
    assert yolo_dataset.classes[0] == 'person'
    assert yolo_dataset.classes[1] == 'bicycle'
