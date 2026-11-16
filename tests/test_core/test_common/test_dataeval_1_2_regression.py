"""Pinned numbers for the DataEval 1.2 changes listed in the CHANGELOG.

Each dataset is small and synthetic, so the expected values can be worked out by hand.
If a DataEval upgrade changes one of them, the CHANGELOG and the analytics reference
need updating along with the test.
"""

import numpy as np
import pytest
import torch
from modelmaite.object_detection import DetectionTarget

from checkmaite.core.image_classification import DataevalCleaning as ICDataevalCleaning
from checkmaite.core.object_detection import DataevalCleaning as ODDataevalCleaning


class _FlatImages:
    """Image-classification dataset of single-value 3x16x16 images."""

    def __init__(self, values, labels, index2label, *, as_float=False):
        self.values = values
        self.labels = labels
        self.as_float = as_float
        self.metadata = {"id": f"flat-{values}-{labels}-{as_float}", "index2label": index2label}

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        image = np.full((3, 16, 16), self.values[i], dtype=np.uint8)
        if self.as_float:
            image = image.astype(np.float32) / 255
        return image, np.eye(len(self.metadata["index2label"]))[self.labels[i]], {"id": i}


@pytest.mark.parametrize("as_float", [False, True], ids=["uint8", "float-0-1"])
def test_brightness_is_on_the_0_255_scale(as_float):
    run = ICDataevalCleaning().run(
        datasets=[_FlatImages([51, 51, 102, 102], [0, 1, 0, 1], {0: "a", 1: "b"}, as_float=as_float)]
    )

    vis = run.outputs.image_stats.vis_stats
    np.testing.assert_allclose(vis.brightness, [51, 51, 102, 102], rtol=1e-5)
    np.testing.assert_allclose(vis.darkness, [51, 51, 102, 102], rtol=1e-5)
    np.testing.assert_allclose(vis.percentiles[0], [51] * 5, rtol=1e-5)
    assert run.extract()[0].mean_brightness == pytest.approx(76.5)


def test_class_count_includes_classes_without_labels():
    # Class "c" is in index2label but no image has it.
    run = ICDataevalCleaning().run(datasets=[_FlatImages([10, 20, 30, 40], [0, 1, 0, 1], {0: "a", 1: "b", 2: "c"})])
    record = run.extract()[0]

    assert dict(run.outputs.label_stats.image_counts_per_class) == {0: 2, 1: 2, 2: 0}
    assert record.class_count == 3
    # The imbalance figures use only the classes that appear, so they stay finite.
    assert record.class_imbalance_ratio == 1.0
    assert record.min_class_image_count == 2
    assert record.max_class_image_count == 2


@pytest.mark.parametrize(("image_count", "expected_outliers"), [(19, 0), (20, 1)])
def test_outliers_need_at_least_20_values(image_count, expected_outliers):
    values = [100 + i % 5 for i in range(image_count)]
    values[0] = 250  # far more than 3 standard deviations from the rest
    run = ICDataevalCleaning().run(
        datasets=[_FlatImages(values, [i % 2 for i in range(image_count)], {0: "a", 1: "b"})]
    )

    assert run.extract()[0].image_outlier_count == expected_outliers
    if expected_outliers:
        assert run.outputs.image_outliers == {0: {"brightness": 250.0, "darkness": 250.0}}


class _BoxesInTwoImageSizes:
    """Two images of different sizes, with boxes at known positions."""

    metadata = {"id": "boxes-in-two-image-sizes", "index2label": {0: "a"}}
    sizes = [(100, 200), (50, 40)]  # (height, width)
    boxes = [[[20.0, 10.0, 70.0, 35.0]], [[4.0, 5.0, 24.0, 30.0], [0.0, 0.0, 40.0, 50.0]]]  # x0, y0, x1, y1

    def __len__(self):
        return 2

    def __getitem__(self, i):
        height, width = self.sizes[i]
        boxes = torch.tensor(self.boxes[i])
        target = DetectionTarget(
            boxes=boxes, labels=torch.zeros(len(boxes), dtype=torch.int64), scores=torch.ones(len(boxes))
        )
        return np.zeros((3, height, width), dtype=np.uint8), target, {"id": i}


def test_box_ratios_are_relative_to_their_own_image():
    run = ODDataevalCleaning().run(datasets=[_BoxesInTwoImageSizes()])
    ratios = run.outputs.box_stats.ratio_stats

    # Box 1 is 50x25 at (20, 10) in a 200x100 image; box 2 is 20x25 at (4, 5) in a
    # 40x50 image; box 3 covers its 40x50 image.
    np.testing.assert_allclose(ratios.offset_x, [0.1, 0.1, 0.0])
    np.testing.assert_allclose(ratios.offset_y, [0.1, 0.1, 0.0])
    np.testing.assert_allclose(ratios.width, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(ratios.height, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(ratios.size, [0.0625, 0.25, 1.0])
