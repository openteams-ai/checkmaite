import numpy as np
import pytest
import torch
from maite.protocols import DatasetMetadata, DatumMetadata
from maite.protocols import object_detection as od

from jatic_ri.core.object_detection.dataeval_feasibility_capability import (
    DataevalFeasibility,
    DataevalFeasibilityConfig,
    _extract_instance_crops,
)
from jatic_ri.core.object_detection.dataset_loaders import DetectionTarget
from jatic_ri.core.report._gradient import HAS_GRADIENT


class FeasibilityTestDataset(od.Dataset):
    """Test dataset with controlled properties for feasibility testing."""

    def __init__(
        self,
        num_images: int = 3,
        num_boxes_per_image: int = 2,
        image_size: tuple[int, int] = (100, 100),
        num_classes: int = 3,
    ):
        self._num_images = num_images
        self._num_boxes = num_boxes_per_image
        self._image_size = image_size
        self._num_classes = num_classes

        self._index2label = {i: f"class_{i}" for i in range(num_classes)}

        self.metadata = DatasetMetadata(
            id="feasibility_test_dataset",
            index2label=self._index2label,
        )

    def __len__(self) -> int:
        return self._num_images

    def __getitem__(self, idx: int) -> tuple:
        h, w = self._image_size

        # Create a simple image with some variation
        image = torch.ones((3, h, w), dtype=torch.uint8) * (50 + idx * 20)
        # Add some patterns to differentiate regions
        image[:, : h // 2, :] += 50
        image[:, :, : w // 2] += 30

        # Create boxes that are well within the image
        box_size = min(h, w) // 4
        boxes = []
        labels = []

        for i in range(self._num_boxes):
            x1 = 10 + i * (box_size + 5)
            y1 = 10 + i * (box_size + 5)
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Ensure boxes are within bounds
            x2 = min(x2, w - 1)
            y2 = min(y2, h - 1)

            boxes.append([x1, y1, x2, y2])
            labels.append((idx + i) % self._num_classes)

        target = DetectionTarget(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.int32),
            scores=torch.ones(len(boxes), dtype=torch.float32),
        )

        datum_metadata = DatumMetadata(id=f"image_{idx}")

        return image, target, datum_metadata


class TestExtractInstanceCrops:
    def test_basic_extraction(self):
        image = np.ones((3, 100, 100), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 40, 40], [50, 50, 80, 80]])

        crops = _extract_instance_crops(
            image=image,
            boxes=boxes,
            context_fraction=0.0,
            resize_size=32,
            min_crop_size=8,
        )

        assert len(crops) == 2
        assert all(crop.shape == (3, 32, 32) for crop in crops)

    def test_context_padding(self):
        image = np.ones((3, 100, 100), dtype=np.uint8) * 128
        boxes = np.array([[20, 20, 50, 50]])  # 30x30 box

        crops_no_ctx = _extract_instance_crops(
            image=image,
            boxes=boxes,
            context_fraction=0.0,
            resize_size=32,
            min_crop_size=8,
        )

        crops_with_ctx = _extract_instance_crops(
            image=image,
            boxes=boxes,
            context_fraction=0.2,
            resize_size=32,
            min_crop_size=8,
        )

        assert len(crops_no_ctx) == 1
        assert len(crops_with_ctx) == 1
        assert crops_no_ctx[0].shape == crops_with_ctx[0].shape

    def test_skip_tiny_boxes(self):
        image = np.ones((3, 100, 100), dtype=np.uint8) * 128
        boxes = np.array(
            [
                [10, 10, 15, 15],  # 5x5 - should be skipped
                [20, 20, 50, 50],  # 30x30 - should be kept
            ]
        )

        crops = _extract_instance_crops(
            image=image,
            boxes=boxes,
            context_fraction=0.0,
            resize_size=32,
            min_crop_size=10,
        )

        assert len(crops) == 1

    def test_chw_input_format(self):
        image_chw = np.ones((3, 100, 100), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 40, 40]])

        crops = _extract_instance_crops(
            image=image_chw,
            boxes=boxes,
            context_fraction=0.0,
            resize_size=32,
            min_crop_size=8,
        )

        assert len(crops) == 1
        assert crops[0].shape == (3, 32, 32)


class TestDataevalFeasibilityCapability:
    @pytest.fixture
    def test_dataset(self):
        return FeasibilityTestDataset(
            num_images=10,
            num_boxes_per_image=10,
            image_size=(100, 100),
            num_classes=20,
        )

    @pytest.fixture
    def test_config(self):
        return DataevalFeasibilityConfig(
            chunk_size=10,
            embedding_batch_size=2,
            crop_resize_size=32,
            target_embedding_dim=64,
        )

    def test_supports_properties(self):
        capability = DataevalFeasibility()

        from jatic_ri.core.capability_core import Number

        assert capability.supports_datasets == Number.ONE
        assert capability.supports_models == Number.ZERO
        assert capability.supports_metrics == Number.ZERO

    def test_outputs(self, test_dataset, test_config):
        capability = DataevalFeasibility()

        # smoke test
        capability.run(
            use_cache=False,
            datasets=[test_dataset],
            config=test_config,
        )

    @pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
    def test_collect_report_consumables(self, test_dataset, test_config):
        capability = DataevalFeasibility()

        output = capability.run(
            use_cache=False,
            datasets=[test_dataset],
            config=test_config,
        )

        slides = output.collect_report_consumables(threshold=0.5)

        assert isinstance(slides, list)

    def test_collect_md_report(self, test_dataset, test_config):
        capability = DataevalFeasibility()

        output = capability.run(
            use_cache=False,
            datasets=[test_dataset],
            config=test_config,
        )

        md_report = output.collect_md_report(threshold=0.5)

        assert isinstance(md_report, str)


class TestDatasetHealthChecks:
    """Tests for dataset health check functionality."""

    def _make_dataset(self, items):
        """Create a minimal dataset from a list of (image, boxes, labels) tuples."""

        class _Dataset(od.Dataset):
            def __init__(self, items):
                self._items = items
                self.metadata = DatasetMetadata(
                    id="health_test",
                    index2label={i: f"cls_{i}" for i in range(20)},
                )

            def __len__(self):
                return len(self._items)

            def __getitem__(self, idx):
                img, boxes, labels = self._items[idx]
                target = DetectionTarget(
                    boxes=torch.tensor(boxes, dtype=torch.float32),
                    labels=torch.tensor(labels, dtype=torch.int32),
                    scores=torch.ones(len(boxes), dtype=torch.float32),
                )
                return img, target, DatumMetadata(id=f"img_{idx}")

        return _Dataset(items)

    def test_single_class_raises(self):
        """Dataset with only 1 class should raise ValueError."""
        items = []
        for _i in range(5):
            img = torch.ones((3, 100, 100), dtype=torch.uint8) * 128
            boxes = [[10, 10, 50, 50]]
            labels = [0]
            items.append((img, boxes, labels))

        dataset = self._make_dataset(items)
        capability = DataevalFeasibility()
        config = DataevalFeasibilityConfig(chunk_size=10, embedding_batch_size=2, crop_resize_size=32)

        with pytest.raises(ValueError, match="at least 2 classes"):
            capability.run(use_cache=False, datasets=[dataset], config=config)

    def test_health_stats_small_objects(self):
        """Dataset with tiny boxes should report small_object_ratio > 0."""
        items = []
        for i in range(6):
            img = torch.ones((3, 200, 200), dtype=torch.uint8) * (50 + i * 30)
            # One tiny box (< 32 max side) and one normal box
            boxes = [[10, 10, 20, 20], [50, 50, 120, 120]]
            labels = [i % 3, (i + 1) % 3]
            items.append((img, boxes, labels))

        dataset = self._make_dataset(items)
        capability = DataevalFeasibility()
        config = DataevalFeasibilityConfig(
            chunk_size=10,
            embedding_batch_size=2,
            crop_resize_size=32,
            target_embedding_dim=64,
        )

        output = capability.run(use_cache=False, datasets=[dataset], config=config)
        assert output.outputs.health_stats.small_object_ratio > 0

    def test_health_stats_truncated_boxes(self):
        """Boxes touching image boundary should report truncated_bbox_ratio > 0."""
        items = []
        for i in range(6):
            img = torch.ones((3, 100, 100), dtype=torch.uint8) * (50 + i * 30)
            # One boundary-touching box and one interior box
            boxes = [[0, 10, 40, 50], [30, 30, 70, 70]]
            labels = [i % 3, (i + 1) % 3]
            items.append((img, boxes, labels))

        dataset = self._make_dataset(items)
        capability = DataevalFeasibility()
        config = DataevalFeasibilityConfig(
            chunk_size=10,
            embedding_batch_size=2,
            crop_resize_size=32,
            target_embedding_dim=64,
        )

        output = capability.run(use_cache=False, datasets=[dataset], config=config)
        assert output.outputs.health_stats.truncated_bbox_ratio > 0

    def test_health_stats_overlapping_boxes(self):
        """Near-identical box pairs should report overlap_image_ratio > 0."""
        items = []
        for i in range(6):
            img = torch.ones((3, 200, 200), dtype=torch.uint8) * (50 + i * 30)
            # Two nearly identical boxes (IoU > 0.85)
            boxes = [[10, 10, 80, 80], [11, 11, 81, 81], [100, 100, 170, 170]]
            labels = [i % 3, (i + 1) % 3, (i + 2) % 3]
            items.append((img, boxes, labels))

        dataset = self._make_dataset(items)
        capability = DataevalFeasibility()
        config = DataevalFeasibilityConfig(
            chunk_size=10,
            embedding_batch_size=2,
            crop_resize_size=32,
            target_embedding_dim=64,
        )

        output = capability.run(use_cache=False, datasets=[dataset], config=config)
        assert output.outputs.health_stats.overlap_image_ratio > 0
