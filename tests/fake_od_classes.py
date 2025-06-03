from collections.abc import Sequence
from typing import Any

import maite.protocols.object_detection as od
import torch
from maite.protocols import ArrayLike, DatasetMetadata, DatumMetadata, MetricMetadata, ModelMetadata
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from jatic_ri.object_detection.datasets import DetectionTarget
from tests.fake_ic_classes import RestartingIterator

"""
The default fake "dataset" will include only one image/target initally.  It is based off of http://cocodataset.org/#explore?id=37777
The image is also saved in this repository in /tests/testing_utilities/example_data/coco_dataset/000000037777.jpg

The characteristics of this fake dataset's repeating target are:

    image (torch.Tensor): the dimensions match the real image but the pixel values here are all "1"
    ObjectDetectionTarget: three boxes of three different classes in the COCO 2017 dataset
    metadata

  The dataset length is set at 6 for now because that was the largest configured size amongst previous fake data implementations.
"""

DEFAULT_DATASET_LENGTH: int = 6

DEFAULT_OD_DATASET_IMAGES: Sequence[ArrayLike] = [
    torch.ones(torch.Size([3, 230, 352]), dtype=torch.uint8) for _ in range(DEFAULT_DATASET_LENGTH)
]

DEFAULT_OD_DATASET_TARGETS: Sequence[od.ObjectDetectionTarget] = [
    DetectionTarget(
        boxes=torch.Tensor(
            [
                [301.84, 74.94, 351.46, 226.38],  # date
                [137.47, 124.11, 197.65, 195.13],  # cherry
                [79.55, 178.05, 287.91, 226.75],  # banana
            ]
        ),
        labels=torch.tensor([4, 3, 2], dtype=torch.int32),
        scores=torch.Tensor([1.0, 1.0, 1.0]),
    )
    for _ in range(DEFAULT_DATASET_LENGTH)
]

# 'Dataset metadata' apply to the entire dataset (e.g. the index2label dictionary)
DEFAULT_OD_DATASET_METADATA = DatasetMetadata(
    id="fake_od_dataset",
    index2label={0: "ignored regions", 1: "apple", 2: "banana", 3: "cherry", 4: "date", 5: "eggplant"},
)

# 'Datum metadata' apply to an individual item within the dataset (e.g. an image ID)
DEFAULT_OD_DATUM_METADATA: Sequence[DatumMetadata] = [
    # RealLabel test stage cache unit tests fail unless each datum in the dataset has a metadata
    # ID which is a string.  A numerical ID cast as a string will fail (at some point it must be converted
    # to int or float and not back).  But the IDs do not have to be unique per image... just any strings.
    DatumMetadata(id="some_string")
    for _ in range(DEFAULT_DATASET_LENGTH)
]

"""
The default OD model returns the same value for each image.  The response is a partially-correct
prediction to the default dataset's image/target data (i.e. six instances of the same image).
Specific deviations from prediction and "ground truth" annotated in comments in-line below.
"""

# 'Model metadata' apply to the entire model (e.g. the index2label dictionary)
DEFAULT_OD_MODEL_METADATA = ModelMetadata(
    id="fake_od_model",
    index2label={0: "ignored regions", 1: "apple", 2: "banana", 3: "cherry", 4: "date", 5: "eggplant"},
)

DEFAULT_OD_MODEL_PREDICTIONS: Sequence[od.ObjectDetectionTarget] = [
    DetectionTarget(
        boxes=torch.Tensor(
            [
                [290, 75, 340, 210],  # accurate bbox, correct label, high confidence
                [100, 50, 110, 100],  # inaccurate bbox, incorrect label, low confidence
                [79.55, 178.05, 287.91, 226.75],  # accurate bbox, incorrect label, high confidence
            ]
        ),
        labels=torch.tensor([4, 1, 5], dtype=torch.int32),
        scores=torch.Tensor([0.9, 0.2, 0.7]),
    )
]

DEFAULT_OD_METRIC_METADATA: MetricMetadata = MetricMetadata(
    id="fake_od_metric",
)
# each value must (1) be safely cast to a float, and (2) possess <value>.numpy() method
DEFAULT_OD_METRIC_RESPONSE: dict[str, Any] = {
    "fake_metric": torch.Tensor([0.12]),
    "fake_metric_1": torch.Tensor([0.43]),
    "fake_metric_2": torch.Tensor([0.56]),
}
# Take the first ordered key above for use as return_key, the 'primary metric' in RI conventions
DEFAULT_OD_METRIC_RETURN_KEY: str = list(DEFAULT_OD_METRIC_RESPONSE.keys())[0]


class FakeODModel(od.Model):
    """This MAITE-compliant fake (stub) OD model can be instantiated with canned responses to its __call__ function for testing.
    A restarting iterator class is used so that a fake model can handle an indefinite number of calls of arbitrary batch sizes.

    IMPORTANT - if the default stub values do not meet your testing requirements, consult the RI Team (in GitLab)
    before implementing a custom instance of this class.  As much as possible, we would like to have as many test cases as possible use
    the default attributes/fixture, so expanding the defaults to cover additional scenarios is likely preferable to creating different fake
    data for different scenarios.

    Per RI conventions, our model classes must expose the model API (e.g. torch.nn.Module) as `model` attribute.
    NOTE: the initialized Torchvision `model` does not actually do inference when this wrapper class is called.  Nor does the index2label contained
    in this fake model's metadata align with this model object.

    Attributes:

        model_metadata (ModelMetadata): Includes `id` and `index2label`.  `index2label` is not required by MAITE definition but _may_ be for RI tools.
        prediction_data (Sequence[od.ObjectDetectionTarget]): A Sequence of responses to provide, sequentially and repeating, as the model is called.
        The __call__ method will batch these responses into a return batch of the correct size.

            The default value should suffice for most (if not all) test cases.  It is (currently) a single response for all images, which is a partially-correct
            prediction to the default dataset's single image as documented with the DEFAULT_OD_MODEL_PREDICTIONS constant above.

            A fake model can also be pre-loaded with the desired responses to a known dataset order.
    """

    def __init__(
        self,
        model_metadata: ModelMetadata = DEFAULT_OD_MODEL_METADATA,
        prediction_data: Sequence[od.ObjectDetectionTarget] = DEFAULT_OD_MODEL_PREDICTIONS,
    ):
        self.prediction_iter: RestartingIterator = RestartingIterator(prediction_data)
        self.metadata: ModelMetadata = model_metadata
        self.model: torch.nn.Module = self._load_default_od_model()

    def __call__(self, input: Sequence[ArrayLike]) -> Sequence[od.ObjectDetectionTarget]:  # noqa: A002
        return [next(self.prediction_iter) for _ in range(len(input))]

    def _load_default_od_model(self) -> torch.nn.Module:
        # NOTE: This model is for unit tests that require accessing the underlying model API only.  Calling the FakeODModel object does NOT
        # use this model for inference.
        return fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=6)


class FakeODDataset(od.Dataset):
    """
    This MAITE-compliant fake OD dataset can be instantiated with any MAITE-compliant representations of (1) images, (2) corresponding ObjectDetectionTarget
    ground truths, and (3) corresponding metadata.

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI Team (in GitLab) before implementing a custom instance of
    this class.  As much as possible, we would like to have as many test cases as possible use the default attributes/fixture, so expanding the defaults to
    cover additional scenarios is likely preferable to creating different fake data for different scenarios.

    The user can make pre-seed their own dataset with any MAITE-compliant data.

    Attributes:
        images (Sequence[ArrayLike]): A Sequence (e.g. list) of images represented in an ArrayLike format (e.g. a torch.Tensor).
        targets (Sequence[od.ObjectDetectionTarget]): A Sequence of targets, one per image consisting of 0 or more boxes, associated classes, and associated scores.
        datum_metadata (Sequence[DatumMetadata]): A Sequence of DatumMetadata objects (a TypedDict) for each image.
        metadata (DatasetMetadata): This is the metadata that applies to the whole dataset, specifically its 'id' and 'index2label' dict
    """

    def __init__(
        self,
        images: Sequence[ArrayLike] = DEFAULT_OD_DATASET_IMAGES,
        targets: Sequence[od.ObjectDetectionTarget] = DEFAULT_OD_DATASET_TARGETS,
        datum_metadata: Sequence[DatumMetadata] = DEFAULT_OD_DATUM_METADATA,
        dataset_metadata: DatasetMetadata = DEFAULT_OD_DATASET_METADATA,
    ):
        self.images: Sequence[ArrayLike] = images
        self.targets: Sequence[od.ObjectDetectionTarget] = targets
        self.datum_metadata: Sequence[DatumMetadata] = datum_metadata
        self.metadata: DatasetMetadata = dataset_metadata

    def __getitem__(self, ind: int) -> tuple[ArrayLike, od.ObjectDetectionTarget, DatumMetadata]:
        return (self.images[ind], self.targets[ind], self.datum_metadata[ind])

    def __len__(self) -> int:
        return len(self.images)


class FakeODMetric(od.Metric):
    """
    This MAITE-compliant fake OD metric can be instantiatiated with a canned response to its compute() method.  By default, it will return
    the value of constant DEFAULT_OD_METRIC_RESPONSE to any invocation of compute().

    IMPORTANT - if the default fake values do not meet your testing requirements, consult the RI Team (in GitLab) before implementing a custom instance of
    this class.  As much as possible, we would like to have as many test cases as possible use the default attributes/fixture, so expanding the defaults to
    cover additional scenarios is likely preferable to creating different fake data for different scenarios.

    Attributes:
        calculated_metrics (dict[str,Any]): A FakeODMetric instantiated with a value here will return that value whenever compute() is called on the object.
        Conventions (/docs/conventions.md) require each value in the returned dict of compute() must (1) be safely cast to a float, and (2) possess
        <value>.numpy() method. This is not required  in MAITE as of verison 0.6.1.
        return_key (str): The Metric class will have an attribute called return_key which describes the top-level performance metric.

    NOTE: Worth considering whether we should update this "metric" with some computationally trivial behavior that would make 'call' more than a stub.
    For example, it could just count the number of calls to update() it has received.

    NOTE: Conventions (/docs/conventions.md) require each value in the returned dict of compupte() must (1) be safely cast to a float, and (2) possess
    <value>.numpy() method. That is stricter than MAITE.  Should we define our own type?
    """

    def __init__(
        self,
        calculated_metrics: dict[str, Any] = DEFAULT_OD_METRIC_RESPONSE,
        metric_metadata: MetricMetadata = DEFAULT_OD_METRIC_METADATA,
        return_key: str = DEFAULT_OD_METRIC_RETURN_KEY,
    ):
        self.calculated_metrics: dict[str, Any] = calculated_metrics
        self.return_key: str = return_key
        self.metadata: MetricMetadata = metric_metadata

    def update(self, preds: Sequence[od.ObjectDetectionTarget], targets: Sequence[od.ObjectDetectionTarget]) -> None:
        pass

    def compute(self) -> dict[str, Any]:
        # each value must (1) be safely cast to a float, and (2) possess <value>.numpy() method
        return self.calculated_metrics

    def reset(self) -> None:
        pass
