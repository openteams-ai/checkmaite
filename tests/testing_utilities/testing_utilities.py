"""Module containing utility functions that can aid in testing."""
import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from pyspark.sql import DataFrame
import torch
from maite.protocols import ArrayLike
from maite.protocols import object_detection as od

from jatic_ri.object_detection.datasets import DetectionTarget

EXAMPLE_DATA_DIR = Path(os.path.abspath(__file__)).parent / "example_data"


def assert_spark_dataframes_equal(
    actual: DataFrame,
    expected: pd.DataFrame,
    orderby: Optional[Union[str, list[str]]] = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Check whether a SparkDataFrame has the same values as a pandas.DataFrame.

    :param actual: Spark dataframe
    :param expected: The expected dataframe as a Pandas dataframe
    :param orderby: Optional param used to order columns in a dataframe
    """
    expected_sorted = (
        expected.sort_values(orderby).reset_index(drop=True)
        if orderby is not None
        else expected.reset_index(drop=True)
    )
    actual_as_pandas: pd.DataFrame = actual.toPandas()
    actual_pandas_sorted = (
        actual_as_pandas.sort_values(orderby).reset_index(drop=True)
        if orderby
        else actual_as_pandas.reset_index(drop=True)
    )
    # check_dtype=False: to ignore differences between, e.g., np.int64 vs np.int32
    # check_like=True: to ignore column order change from `toPandas`
    return pd.testing.assert_frame_equal(
        actual_pandas_sorted,
        expected_sorted,
        check_dtype=False,
        **kwargs,
    )

def minimal_maite_object_detection_dataset_and_model(dataset_length: int) -> tuple[od.Dataset, od.Model]:
    """
    A factory for creating a lightweight MAITE-compliant object-detection dataset and model for rapid
    testing.

    The dataset has been prepared so as to include a small number of accurate bounding box examples.

    The model returns a canned response, for fast testing. This canned response has been prepared so
    as to include a small number of accurate and inaccurate (relative to the ground-truth) examples.

    Attributes
    ----------
    dataset_length : int
        The total number of data points in the dataset. Each data point will be identical.
    """

    # this text fixture has been prepared relative to the following file located in EXAMPLE_DATA_DIR
    _local_filepath = "coco_dataset"
    _img_filename = "000000037777.jpg"
    _img_shape = torch.Size([3, 230, 352])

    def ground_truth_factory() -> od.ObjectDetectionTarget:
        """Returns detections that can be used as ground-truth for the image."""
        return DetectionTarget(
            boxes=torch.tensor(
                [
                    [301.84, 74.94, 351.46, 226.38],  # fridge
                    [137.47, 124.11, 197.65, 195.13],  # oven
                    [79.55, 178.05, 287.91, 226.75],  # dining table
                ]
            ),
            labels=torch.tensor([82, 79, 67]),
            scores=torch.tensor([1.0, 1.0, 1.0]),
        )

    def prediction_factory() -> od.ObjectDetectionTarget:
        """
        Returns detections that can be used as predictions for the image.

        Detections have been selected so that some bounding boxes are
        predicted accurately, and others are predicted inaccurately. In
        addition, some inaccurate bounding boxes are predicted with high
        confidence, whereas others are with low confidence.
        """
        return DetectionTarget(
            boxes=torch.tensor(
                [
                    [290, 75, 340, 210],  # accurate bbox, correct label, high confidence
                    [100, 50, 110, 100],  # inaccurate bbox, incorrect label, low confidence
                    [79.55, 178.05, 287.91, 226.75],  # accurate bbox, incorrect label, high confidence
                ]
            ),  
            labels=torch.tensor([82, 11, 23]),
            scores=torch.tensor([0.9, 0.2, 0.7]),
        )

    class MinimalMAITEDataset:
        """A lightweight implementation of a MAITE-compliant object-detection dataset for rapid testing."""

        def __init__(self, dataset_path: str) -> None:
            self._dataset_path = dataset_path  # seems to be required by certain JATIC packages...

        def __getitem__(self, ind: int) -> tuple[ArrayLike, od.ObjectDetectionTarget, dict[str, Any]]:
            if ind >= dataset_length:
                raise IndexError(
                    f"The index number {ind} is out of range for the dataset which has length {dataset_length}",
                )

            metadata = {"id": f"image-{ind}", "local_filepath": _local_filepath, "img_filename": _img_filename}

            return (torch.ones(_img_shape, dtype=torch.uint8), ground_truth_factory(), metadata)

        def __len__(self) -> int:
            return dataset_length

    dataset = MinimalMAITEDataset(dataset_path=EXAMPLE_DATA_DIR)

    class MinimalMAITEModel:
        """A lightweight implementation of a MAITE-compliant object-detection model for rapid testing."""

        def __call__(self, input_batch: od.InputBatchType) -> od.TargetBatchType:
            return [prediction_factory() for i in range(len(input_batch))]

    model = MinimalMAITEModel()

    return dataset, model
