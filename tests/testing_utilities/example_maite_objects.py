"""Module containing MAITE-wrapped objects and datasets used during the Increment 5 demo. For use in testing."""
import copy
import os
import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Union

import numpy as np
import pandas as pd
import torch
import yolov5
from maite.protocols import object_detection as od
from maite.protocols.object_detection import TargetBatchType as OD_TargetBatchType
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms.v2 import Resize

to_1280x1280 = Resize((1280, 1280))

EXAMPLE_MODEL_DIR = Path(os.path.abspath(__file__)).parent / "example_models"

YOLOV5S_USA_ALL_SEASONS_V1_MODEL_PATH = EXAMPLE_MODEL_DIR / "yolov5s_USA-All-Seasons_v1.pt"
YOLOV5S_USA_RUS_ALL_SEASONS_V1_MODEL_PATH = EXAMPLE_MODEL_DIR / "yolov5s_USA-RUS-All-Seasons_v1.pt"

EXAMPLE_DATA_DIR = Path(os.path.abspath(__file__)).parent / "example_data"
USA_SUMMER_DATA_IMAGERY_DIR = EXAMPLE_DATA_DIR / "USA_summer"
USA_SUMMER_DATA_METADATA_FILE_PATH = EXAMPLE_DATA_DIR / "USA_summer" / "dev_val.csv"

@dataclass
class Yolov5sDetectionTarget:
    """Yolov5s detection target."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class Yolov5sModel:
    """Yolov5s model."""

    def __init__(self, model_path: str, transforms: Any, device: str):  # noqa: ANN401
        assert Path(model_path).exists(), f"Model file {model_path} does not exist"
        self.model = yolov5.load(str(model_path))
        self.transforms = transforms
        self.device = device

        self.model.eval()
        self.model.to(device)

    def __call__(self, input_bt: od.InputBatchType) -> Sequence[Yolov5sDetectionTarget]:
        """Call."""
        batch = torch.tensor(input_bt)
        assert len(batch.shape) == 4
        _, _, orig_img_height, orig_img_width = batch.shape
        batch = to_1280x1280(batch)

        batch_size, _, _, _ = batch.shape

        # This call is /VERY/ touchy. The return type depends on the parameter type. If
        # changed to torch.tensor, another tensor will be returned. Keep the parameter
        # a np.array so a structured prediction object is returned.
        yolov5_predictions_batch = self.model(
            [np.array(batch[i]).squeeze() for i in range(batch_size)],
        )

        predictions_batch = []
        for predict_datum in yolov5_predictions_batch.xyxy:
            if len(predict_datum) == 0:
                boxes_datum = torch.zeros(0, 4)
                labels_datum = torch.zeros(0, dtype=torch.uint8)
                scores_datum = torch.zeros(0)
            else:
                boxes_datum = torch.zeros(len(predict_datum), 4)
                labels_datum = torch.zeros(len(predict_datum), dtype=torch.uint8)
                scores_datum = torch.zeros(len(predict_datum))
                for i in range(len(predict_datum)):
                    det_box = predict_datum[i]
                    if det_box[0].numel() != 0:
                        boxes_datum[i, :4] = det_box[:4]
                        labels_datum[i] = det_box[5].int()
                        scores_datum[i] = det_box[4]

                boxes_datum = self.normalize_bbox_tensor(
                    boxes_datum,
                    img_width=orig_img_width,
                    img_height=orig_img_height,
                )

            predictions_batch.append(
                Yolov5sDetectionTarget(
                    boxes=boxes_datum.detach().cpu(),  # xmin, ymin, xmax, ymax
                    labels=labels_datum.detach().cpu(),  # class 0, 1..
                    scores=scores_datum.detach().cpu(),  # confidence
                ),
            )

        return predictions_batch

    def normalize_bbox_tensor(self, boxes, *, img_width, img_height) -> torch.Tensor:
        """Normalize box tensor."""
        # YOLO model expects images resized to (1280,1280). We must scale the
        # bounding boxes to match the original image size.
        x_ratio = img_width / 1280
        y_ratio = img_height / 1280
        xmin, ymin, xmax, ymax = (
            boxes[:, 0] * x_ratio,
            boxes[:, 1] * y_ratio,
            boxes[:, 2] * x_ratio,
            boxes[:, 3] * y_ratio,
        )
        return torch.stack([xmin, ymin, xmax, ymax]).T

    @property
    def name(self) -> str:
        """Return name of class"""
        return self.__class__.__name__


@dataclass
class FMOWDetectionTarget:
    """Detection target for FMOW."""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


PER_DATUM_METRIC_KEY = "per_datum"


class FMOWDetectionDataset:
    """
    Maite-wrapped object detection dataset built using image files
    within <dataset_dir> and metadata from file <metadata_filepath>. Rows in
    <metadata_filepath> correspond to individual image files.

    parameters
    ----------
    dataset_dir: Path
        directory containing image files

    metadata_filepath: Path | list[Path]
        fully qualified filepath containing metadata for all images in dataset (or
        a list of such filepaths) in csv format
        This file should have at least the following columns:
            - 'img_filename' containing the filename for the image file associated
               with each image
            - 'class' containing the string-valued class label for each image
            - 'r0', 'c0', 'r1', 'c1' containing the integer-typed pixel coordinates
                of bounding box containing an object of the expected class

    Note: Each image only contains one bounding box in FMOW dataset, so the mapping
    from 'img_path' to class/bounding-box is necessarily one-to-one.
    """

    # labels correspond to category in FMOW data metadata
    labels = [
        "airport_terminal",
        "dam",
        "electric_substation",
        "factory_or_powerplant",
        "hospital",
        "military_facility",
        "nuclear_powerplant",
        "oil_or_gas_facility",
        "place_of_worship",
        "prison",
        "road_bridge",
        "stadium",
    ]

    label2ind = {lbl: ind for ind, lbl in enumerate(labels)}

    def __init__(  # noqa: C901
        self,
        dataset_dir: Path,
        metadata_filepath: Union[Path, list[Path]],
        img_file_ext="jpeg",
    ):

        # if metadata_filepath is a single string, make it into a length-1 list
        if isinstance(metadata_filepath, Path):
            metadata_filepath = [metadata_filepath]

        # load inputs into instance member variables
        self._dataset_path: Path = dataset_dir
        self._metadata_filepaths: list[Path] = metadata_filepath

        # check existence of dataset_dir
        assert (
            self._dataset_path.is_dir()
        ), f"provided dataset_dir {dataset_dir} doesn't exist"

        # check existence of metadata file(s) and check csv file extension(s)
        for md_path in self._metadata_filepaths:
            if not md_path.exists() or md_path.suffix != ".csv":
                raise ValueError(
                    f"provided metadata {metadata_filepath} must be a csv file on disk",
                )

        # read metadata_paths into a pandas dataframe
        md_dfs: list[pd.DataFrame] = [
            pd.read_csv(md_path) for md_path in self._metadata_filepaths
        ]

        self._metadata_df = pd.concat(md_dfs, copy=False, join="inner")

        # check for required keys in _metadata_df
        required_md_keys = ("img_filename", "r0", "r1", "c0", "c1")
        missing_md_keys = [
            key for key in required_md_keys if key not in self._metadata_df.columns
        ]
        if len(missing_md_keys) > 0:
            required_md_key_str = "\n".join(missing_md_keys)
            raise RuntimeError(
                f"Metadata csv is missing the following required key(s):\n {required_md_key_str}",
            )

        self._metadata_df = self._metadata_df.set_index(
            "img_filename",
            drop=False,
        )  # permit easy metadata lookups via filename

        # Retain only the metadata for images present locally, then
        # add a 'local_filepath' column to dataframe containing the path to
        # each file relative to self._dataset_path

        full_image_filepaths = sorted(  # noqa: C414
            list(self._dataset_path.glob(f"**/*{img_file_ext}")),
        )
        local_image_paths = [
            str(p.relative_to(self._dataset_path).parent) for p in full_image_filepaths
        ]
        local_filenames = [p.name for p in full_image_filepaths]

        # Removing local filenames from dataset that aren't
        # present in metadata (so we don't have issues when we try to downselect
        # metadata based on available files) -- we warn user this is happening
        # (metafiles may be augmented to negate the need for this step)
        # --------
        full_image_filepaths = [
            ff
            for ff, fn in zip(full_image_filepaths, local_filenames)
            if fn in self._metadata_df.index
        ]

        local_image_paths = [
            ip
            for ip, fn in zip(local_image_paths, local_filenames)
            if fn in self._metadata_df.index
        ]

        data_missing_md = set(local_filenames) - {p.name for p in full_image_filepaths}

        local_filenames = [
            fn for fn in local_filenames if fn in self._metadata_df.index
        ]

        if len(data_missing_md) > 0:
            warnings.warn(
                f"There were {len(data_missing_md)} files on disk that don't have associated metadata.\n"
                f"They will not be represented in the constructed dataset.",
                stacklevel=2,
            )

        # --------

        # verify metadata file contains all paths that were found
        # (Note: this may be a redundant check)
        imgfiles_missing_md = [
            p.name
            for p in full_image_filepaths
            if p.name not in self._metadata_df.index
        ]

        if len(imgfiles_missing_md) > 0:
            max_print_lines = 20
            missing_fname_str = "\n".join(imgfiles_missing_md[:max_print_lines])
            if len(imgfiles_missing_md) > max_print_lines:
                missing_fname_str += "\n...(and others)..."
            warnings.warn(
                f"The following {len(imgfiles_missing_md)} image files were not "
                "found in metadata file:\n {missing_fname_str}",
                stacklevel=2,
            )

        # trim metadata to only include files present on disk
        files_not_present = set(self._metadata_df.index.values) - set(local_filenames)
        if len(files_not_present) > 0:
            warnings.warn(
                f"There were {len(files_not_present)} in metadata that weren't found on disk",
                stacklevel=2,
            )
            self._metadata_df = self._metadata_df.loc[
                local_filenames
            ]  # trim metadata down to what is available locally

        # add local_filepath information to metadata for easy lookup
        local_filepath_series = pd.Series(local_image_paths, index=local_filenames)

        # (a) remove duplicate indices from local_filepath_series by keeping the first occurrence
        local_filepath_series_unique = local_filepath_series.loc[
            ~local_filepath_series.index.duplicated(keep="first")
        ]
        local_filepath_series_unique.name = "local_filepath"

        # (b) join the DataFrames on their indices
        self._metadata_df = self._metadata_df.join(
            local_filepath_series_unique,
            how="left",
        )

        # check that class labels provided by metadata are all elements of
        # the 'labels' class variable:
        # (Confusingly, 'label' is found in metadata file under 'class' heading
        # we avoid 'class' term here for clarity)
        labels_from_md = sorted(list(self._metadata_df["class"].unique()))  # noqa: C414
        unexpected_md_labels = set(labels_from_md) - set(FMOWDetectionDataset.labels)
        if len(unexpected_md_labels) > 0:
            unexpected_label_str = "\n".join(unexpected_md_labels)
            expected_label_str = "\n".join(FMOWDetectionDataset.labels)
            raise RuntimeError(
                f"Unexpected data labels encountered in metadata.\n"
                f"Unexpected labels:\n {unexpected_label_str}\n"
                f"Expected label list:\n {expected_label_str}",
            )

        self._bad_img_filepaths: list[Path] = self._check_dataset_image_loading()
        if len(self._bad_img_filepaths) > 0:
            img_names = [fp.name for fp in self._bad_img_filepaths]
            self._metadata_df = self._metadata_df.drop(index=img_names)
            print(
                f"removed {len(self._bad_img_filepaths)} from dataset due to issues loading from disk",
            )

    def _check_dataset_image_loading(self) -> list[Path]:
        """Try to lazily load data from all images in dataset.
        Remove any elements that fail with UnidentifiedImageError"""

        import PIL

        bad_img_filepaths: list[Path] = []
        for _, srs in self._metadata_df.iterrows():
            # get original data item
            image_path = (
                self._dataset_path
                / Path(srs["local_filepath"])
                / Path(srs["img_filename"])
            )

            try:
                Image.open(image_path)
            except PIL.UnidentifiedImageError:
                bad_img_filepaths.append(image_path)

        # only warn if there were any 'bad' image paths
        if len(bad_img_filepaths) > 0:
            warnings.warn(
                f"Found {len(bad_img_filepaths)} "
                "img files from dataset with issues loading from file...",
                stacklevel=2,
            )

        # return 'bad' image paths
        return bad_img_filepaths

    def __len__(self) -> int:
        """Length of dataset"""
        return len(self._metadata_df)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[np.ndarray, FMOWDetectionTarget, dict[str, Any]]:
        """Get item from dataset"""
        if index < len(self):
            # get original data item
            image_path = (
                self._dataset_path
                / Path(self._metadata_df.iloc[index]["local_filepath"])
                / Path(self._metadata_df.iloc[index]["img_filename"])
            )

            image = Image.open(image_path)
            # format input as numpy array, reshape it to be CxHxW
            img_np = np.moveaxis(np.asarray(image), 2, 0)

            if isinstance(self._metadata_df.loc[image_path.name], pd.Series):
                # single matching row
                metadatas = [dict(self._metadata_df.loc[image_path.name])]
            else:
                # multiple matching rows
                metadatas = [
                    dict(row)
                    for _, row in self._metadata_df.loc[image_path.name].iterrows()
                ]

            bboxes_list: list[torch.Tensor] = (
                []
            )  # accumulate bounding x0,y0,x1,y1 bounding
            # boxes in shape (1,4) to be concatenated afterward
            labels_list: list[torch.Tensor] = (
                []
            )  # accumulate (1,Cl) onehot vectors to be concatenated
            # afterwards

            for metadata in metadatas:
                # get target and metadata

                label = metadata["class"]
                assert isinstance(label, str), "label expected to be a string value"
                for key, value in metadata.items():
                    try:
                        float_value = float(str(value))
                        metadata[key] = float_value
                        continue
                    except (ValueError, TypeError, KeyError):
                        pass

                    if value == "True" or isinstance(value, type(np.bool_(True))):
                        metadata[key] = True
                    elif value == "False" or isinstance(value, type(np.bool_(False))):
                        metadata[key] = False

                x0, x1, y0, y1 = (
                    metadata["r0"],
                    metadata["r1"],
                    metadata["c0"],
                    metadata["c1"],
                )
                bboxes_list.append(torch.Tensor([x0, y0, x1, y1]).unsqueeze(0))
                labels_list.append(
                    torch.tensor([FMOWDetectionDataset.label2ind[label]]),
                )  # torch.Tensor([1]) will create a dtype=32 tensor

            target = FMOWDetectionTarget(
                boxes=torch.cat(bboxes_list, dim=0),
                labels=torch.cat(labels_list, dim=0),
                scores=torch.Tensor([1] * len(bboxes_list)),
            )

        else:
            raise IndexError("Index out of bounds")

        return img_np, target, metadata


class WrappedODMetric:
    """Wraps an object detection metric."""

    def __init__(
        self,
        od_metric: Callable[
            [list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]],
            dict[str, Any],
        ],
        return_key: str,  # single key to return from Torchmetric's return dict
    ):
        self._od_metric = od_metric
        # For why this copy is needed, see the comment in update()
        self._od_metric_by_datum = copy.deepcopy(od_metric)
        self._return_key = return_key
        self.results = {PER_DATUM_METRIC_KEY: {self._return_key: []}}

    # create utility function to convert ObjectDetectionTarget_impl type to what
    # the type expected by torchmetrics IntersectionOverUnion metric
    @staticmethod
    def to_tensor_dict(tgt: od.ObjectDetectionTarget) -> dict[str, torch.Tensor]:
        """
        Convert an ObjectDetectionTarget_impl into a dictionary expected internally by
        raw `update` method of raw torchmetrics method
        """
        return {
            "boxes": torch.as_tensor(tgt.boxes),
            "scores": torch.as_tensor(tgt.scores),
            "labels": torch.as_tensor(tgt.labels),
        }

    def update(
        self,
        preds: OD_TargetBatchType,
        targets: OD_TargetBatchType,
    ) -> None:
        # convert to natively-typed from of preds/targets
        preds_tm = [self.to_tensor_dict(pred) for pred in preds]
        targets_tm = [self.to_tensor_dict(tgt) for tgt in targets]
        self._od_metric.update(preds_tm, targets_tm)

    def compute(self) -> dict[str, Any]:
        all_results = self._od_metric.compute()
        assert self._return_key in all_results, f"key '{self._return_key}' not in Torchmetrics results"
        return {self._return_key: all_results[self._return_key]}

    def reset(self) -> None:
        self._od_metric.reset()


def create_maite_wrapped_metric(name: Literal["mAP_50"]) -> od.Metric:
    if name == "mAP_50":
        tm_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5],
            rec_thresholds=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            max_detection_thresholds=[1, 10, 100],
            class_metrics=False,
            extended_summary=False,
            average="macro",
        )
        return WrappedODMetric(tm_metric, return_key="map_50")
    raise Exception(f"Unsupported object detection metric: {name}")


def cleanup_test_residue(img_path: Path, cache_dir: Path) -> None:
    """Clean up generated content

    Args:
        img_path (Path): path to image to delete
        cache_dir (Path): path to cache dir
    """
    # Clean up cache
    if len(os.listdir(cache_dir.parent)) == 0:
        # delete .tscache if no other files are present
        shutil.rmtree(cache_dir.parent, ignore_errors=True)
    else:
        # otherwise, just delete reallabel cache
        shutil.rmtree(cache_dir, ignore_errors=True)
    # Clean up created images
    shutil.rmtree(img_path.parent, ignore_errors=True)
