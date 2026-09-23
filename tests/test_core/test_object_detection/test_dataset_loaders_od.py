"""Tests for checkmaite's native-datamaite OD loading boundary."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from datamaite.object_detection import ObjectDetectionDataset
from upath import UPath

from checkmaite.core._common.dataset_utils import (
    DatasetSourceError,
    enforce_source_integrity,
    to_caller_anchored_path,
)
from checkmaite.core.object_detection.dataset_loaders import (
    YoloDetectionDataLoader,
    _with_dataset_id,
    load_coco_detection_dataset,
    load_datasets,
    load_visdrone_detection_dataset,
    load_yolo_detection_dataset,
)

ROOT = Path(__file__).parents[2] / "data_for_tests"
COCO_ROOT = ROOT / "coco_dataset"
YOLO_ROOT = ROOT / "yolo_dataset"
VISDRONE_ROOT = ROOT / "visdrone_dataset"


def _assert_native_maite_item(dataset: ObjectDetectionDataset) -> None:
    image, target, metadata = dataset[0]
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[0] == 3
    assert isinstance(target.boxes, np.ndarray)
    assert isinstance(target.labels, np.ndarray)
    assert isinstance(target.scores, np.ndarray)
    assert target.boxes.shape[-1] == 4
    assert "id" in metadata


def test_coco_returns_native_datamaite_dataset() -> None:
    dataset = load_coco_detection_dataset(COCO_ROOT, str(COCO_ROOT / "ann_file.json"), dataset_id="coco-test")
    assert type(dataset) is ObjectDetectionDataset
    assert dataset.metadata["id"] == "coco-test"
    assert len(dataset) == 4
    _assert_native_maite_item(dataset)


def test_visdrone_returns_native_datamaite_dataset() -> None:
    dataset = load_visdrone_detection_dataset(VISDRONE_ROOT, dataset_id="visdrone-test")
    assert type(dataset) is ObjectDetectionDataset
    assert dataset.metadata["id"] == "visdrone-test"
    assert len(dataset) == 3
    _assert_native_maite_item(dataset)


def test_yolo_returns_only_requested_split_as_native_dataset() -> None:
    dataset = load_yolo_detection_dataset(YOLO_ROOT / "dataset.yaml", split="train", dataset_id="yolo-test")
    assert type(dataset) is ObjectDetectionDataset
    assert dataset.metadata["id"] == "yolo-test"
    assert len(dataset) == 4
    _assert_native_maite_item(dataset)


def test_native_dataset_provides_maite_fieldwise_access() -> None:
    # datamaite >=0.4.0 implements the MAITE fieldwise accessors natively;
    # checkmaite intentionally has no adapter layer re-adding them.
    dataset = load_coco_detection_dataset(COCO_ROOT, str(COCO_ROOT / "ann_file.json"))
    image, target, metadata = dataset[0]
    np.testing.assert_array_equal(dataset.get_input(0), image)
    np.testing.assert_array_equal(dataset.get_target(0).boxes, target.boxes)
    np.testing.assert_array_equal(dataset.get_target(0).labels, target.labels)
    assert dataset.get_metadata(0) == metadata


def test_yolo_ann_dir_override_loads_labels(tmp_path: Path) -> None:
    # datamaite >=0.4.0 accepts an ann_dir loader option; no staged symlink tree.
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "custom_labels").mkdir()
    for image in (YOLO_ROOT / "images").iterdir():
        (root / "images" / image.name).write_bytes(image.read_bytes())
    for label in (YOLO_ROOT / "labels").iterdir():
        (root / "custom_labels" / label.name).write_bytes(label.read_bytes())
    (root / "dataset.yaml").write_text((YOLO_ROOT / "dataset.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    dataset = load_yolo_detection_dataset(root / "dataset.yaml", ann_dir=root / "custom_labels")
    assert len(dataset) == 4
    assert dataset.num_detections > 0


def test_yolo_arbitrary_yaml_name_is_supported(tmp_path: Path) -> None:
    # datamaite >=0.4.0 accepts an explicit yaml_file option, so callers are
    # not restricted to the conventional data.yaml/dataset.yaml names.
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for image in (YOLO_ROOT / "images").iterdir():
        (root / "images" / image.name).write_bytes(image.read_bytes())
    for label in (YOLO_ROOT / "labels").iterdir():
        (root / "labels" / label.name).write_bytes(label.read_bytes())
    (root / "custom-name.yaml").write_text((YOLO_ROOT / "dataset.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    dataset = load_yolo_detection_dataset(root / "custom-name.yaml")
    assert len(dataset) == 4
    assert dataset.num_detections > 0


def test_load_datasets_dispatches_by_format_not_class_name() -> None:
    loaded = load_datasets(
        {
            "evaluation": {
                "dataset_format": "coco",
                "data_dir": str(COCO_ROOT),
                "metadata_path": str(COCO_ROOT / "ann_file.json"),
            }
        }
    )
    assert type(loaded["evaluation"]) is ObjectDetectionDataset


def test_batch_loader_accepts_native_maite_dataset() -> None:
    dataset = load_yolo_detection_dataset(YOLO_ROOT / "dataset.yaml")
    loader = YoloDetectionDataLoader(dataset, batch_size=3)
    batches = list(loader)
    assert len(batches) == 2
    assert sum(len(inputs) for inputs, _targets, _metadata in batches) == len(dataset)


def _copy_yolo(root: Path, *, labels_dir: str = "labels") -> Path:
    """Copy the YOLO fixture under ``root``, optionally renaming ``labels/``."""
    (root / "images").mkdir(parents=True)
    (root / labels_dir).mkdir(exist_ok=True)
    for image in (YOLO_ROOT / "images").iterdir():
        (root / "images" / image.name).write_bytes(image.read_bytes())
    for label in (YOLO_ROOT / "labels").iterdir():
        (root / labels_dir / label.name).write_bytes(label.read_bytes())
    (root / "dataset.yaml").write_text((YOLO_ROOT / "dataset.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return root


def _copy_coco(root: Path) -> Path:
    root.mkdir(parents=True)
    for item in COCO_ROOT.iterdir():
        if item.is_file():
            (root / item.name).write_bytes(item.read_bytes())
    return root


class TestCallerRelativeAnnotationPaths:
    """``ann_file``/``ann_dir`` keep their historical caller-relative meaning.

    datamaite anchors a relative override to the dataset root; checkmaite's API
    has always anchored it to the working directory. Without the boundary
    anchoring, a COCO call resolves ``<root>/<root>/ann_file.json`` and a YOLO
    call silently loads every image with zero detections.
    """

    def test_coco_relative_ann_file_resolves_against_cwd(self, tmp_path: Path, monkeypatch) -> None:
        _copy_coco(tmp_path / "data" / "coco")
        monkeypatch.chdir(tmp_path)

        dataset = load_coco_detection_dataset("data/coco", "data/coco/ann_file.json")

        assert len(dataset) == 4
        assert dataset.num_detections == 57

    def test_yolo_relative_ann_dir_resolves_against_cwd(self, tmp_path: Path, monkeypatch) -> None:
        _copy_yolo(tmp_path / "data" / "yolo", labels_dir="custom_labels")
        monkeypatch.chdir(tmp_path)

        dataset = load_yolo_detection_dataset("data/yolo/dataset.yaml", ann_dir="data/yolo/custom_labels")

        assert len(dataset) == 4
        # The failure mode this pins is silent: a misresolved ann_dir still
        # loads all four images, with no detections on any of them.
        assert dataset.num_detections == 56

    def test_yolo_ann_dir_override_loads_every_annotation(self, tmp_path: Path) -> None:
        _copy_yolo(tmp_path / "dataset", labels_dir="custom_labels")

        dataset = load_yolo_detection_dataset(
            tmp_path / "dataset" / "dataset.yaml", ann_dir=tmp_path / "dataset" / "custom_labels"
        )

        assert dataset.num_detections == load_yolo_detection_dataset(YOLO_ROOT / "dataset.yaml").num_detections


class TestMalformedAnnotationsFailLoudly:
    """datamaite drops unparseable rows with a warning; checkmaite raises.

    Converting a rejected annotation into background silently biases every
    detection metric computed from the dataset, so the boundary refuses the
    dataset and reports the offending ``file:line``.
    """

    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            ("0 0.5 0.5 0.2\n", "expected 5 or 6 fields"),
            ("0 0.5 0.5 0.2 0.2 0.5 0.5\n", "expected 5 or 6 fields"),
            ("0 nope 0.5 0.2 0.2\n", "invalid bbox"),
            ("0 0.5 0.5 9.0 0.2\n", "out-of-range normalized bbox"),
            ("0 0.5 0.5 0.0 0.2\n", "out-of-range normalized bbox"),
            ("x 0.5 0.5 0.2 0.2\n", "invalid class id"),
            ("0 0.5 0.5 0.2 0.2 7.0\n", "invalid confidence"),
        ],
    )
    def test_yolo_rejected_row_raises_with_diagnostics(self, tmp_path: Path, row: str, expected: str) -> None:
        root = _copy_yolo(tmp_path / "dataset")
        label = sorted((root / "labels").iterdir())[0]
        label.write_text(row, encoding="utf-8")

        with pytest.raises(DatasetSourceError) as excinfo:
            load_yolo_detection_dataset(root / "dataset.yaml")

        message = str(excinfo.value)
        assert expected in message
        assert f"{label}:1" in message

    def test_partially_malformed_dataset_is_rejected_not_silently_shortened(self, tmp_path: Path) -> None:
        root = _copy_yolo(tmp_path / "dataset")
        label = sorted((root / "labels").iterdir())[0]
        kept, *_ = label.read_text(encoding="utf-8").splitlines()
        label.write_text(f"{kept}\n0 0.5 0.5 0.2\n", encoding="utf-8")

        with pytest.raises(DatasetSourceError, match="1 source record"):
            load_yolo_detection_dataset(root / "dataset.yaml")

    def test_opting_out_returns_the_parseable_records_with_a_warning(self, tmp_path: Path) -> None:
        root = _copy_yolo(tmp_path / "dataset")
        label = sorted((root / "labels").iterdir())[0]
        rejected = len(label.read_text(encoding="utf-8").splitlines())
        label.write_text("0 0.5 0.5 0.2\n" * rejected, encoding="utf-8")
        intact = load_yolo_detection_dataset(YOLO_ROOT / "dataset.yaml").num_detections

        with pytest.warns(UserWarning, match=f"{rejected} source record"):
            dataset = load_yolo_detection_dataset(root / "dataset.yaml", strict_annotations=False)

        assert len(dataset) == 4
        assert dataset.num_detections == intact - rejected

    def test_all_rejected_labels_are_distinguished_from_unlabelled_images(self, tmp_path: Path) -> None:
        # An image with no label file at all is a legitimate background image;
        # an image whose label file was entirely rejected is a data error. The
        # two must not both surface as "zero detections".
        unlabelled = _copy_yolo(tmp_path / "unlabelled")
        for label in (unlabelled / "labels").iterdir():
            label.unlink()
        dataset = load_yolo_detection_dataset(unlabelled / "dataset.yaml")
        assert len(dataset) == 4
        assert dataset.num_detections == 0

        rejected = _copy_yolo(tmp_path / "rejected")
        for label in (rejected / "labels").iterdir():
            label.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
        with pytest.raises(DatasetSourceError):
            load_yolo_detection_dataset(rejected / "dataset.yaml")

    def test_coco_malformed_annotation_is_rejected(self, tmp_path: Path) -> None:
        root = _copy_coco(tmp_path / "coco")
        ann_path = root / "ann_file.json"
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        payload["annotations"][0].pop("bbox")
        ann_path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(DatasetSourceError, match="COCO annotation"):
            load_coco_detection_dataset(root, str(ann_path))

    def test_visdrone_malformed_row_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "visdrone"
        shutil.copytree(VISDRONE_ROOT, root)
        annotation = sorted((root / "annotations").iterdir())[0]
        annotation.write_text("1,2,3\n", encoding="utf-8")

        with pytest.raises(DatasetSourceError, match=r"VisDrone row .*:1"):
            load_visdrone_detection_dataset(root)

    def test_coco_annotation_orphaned_from_images_is_rejected(self, tmp_path: Path) -> None:
        # datamaite reports this loss as "Dropping ...", not "Skipping ...".
        root = _copy_coco(tmp_path / "coco")
        ann_path = root / "ann_file.json"
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        payload["annotations"][0]["image_id"] = 999999  # absent from images[]
        ann_path.write_text(json.dumps(payload), encoding="utf-8")
        intact = load_coco_detection_dataset(COCO_ROOT, str(COCO_ROOT / "ann_file.json")).num_detections

        with pytest.raises(DatasetSourceError, match=r"Dropping 1 COCO annotation"):
            load_coco_detection_dataset(root, str(ann_path))

        with pytest.warns(UserWarning, match="1 source record"):
            dataset = load_coco_detection_dataset(root, str(ann_path), strict_annotations=False)
        assert dataset.num_detections == intact - 1

    def test_unreadable_visdrone_annotation_file_is_rejected(self, tmp_path: Path) -> None:
        # A file datamaite cannot decode at all is reported as "Could not read".
        root = tmp_path / "visdrone"
        shutil.copytree(VISDRONE_ROOT, root)
        annotation = sorted((root / "annotations").iterdir())[0]
        annotation.write_bytes(b"\xff\xfe\x00 not utf-8 \x80\x81")

        with pytest.raises(DatasetSourceError, match="Could not read VisDrone annotation file"):
            load_visdrone_detection_dataset(root)


@pytest.mark.parametrize(
    "message",
    [
        "Dropping 3 COCO annotation(s) referencing 2 image id(s) missing from images[] in a.json",
        "Could not read COCO annotation file a.json: bad JSON",
        "Could not read YOLO label file l.txt: decode error",
        "Could not read VisDrone annotation file v.txt: decode error",
        "Could not list VisDrone frame directory d: permission denied",
        "COCO annotation 7 in a.json has a missing/invalid category_id; it will carry no label",
        "YOLO OD: ann_dir label l.txt is claimed by 2 images (a, b); leaving them unlabelled rather than assign",
        "Skipping labels in l.txt because image dimensions could not be determined",
    ],
)
def test_ground_truth_loss_phrasings_are_recognized(message: str) -> None:
    with pytest.raises(DatasetSourceError):
        enforce_source_integrity([message], source="fixture", strict=True)


@pytest.mark.parametrize(
    "message",
    [
        # The box is kept; only its name is unknown -- a taxonomy mismatch, not lost ground truth.
        "YOLO label row l.txt:3 references class id 9 not defined in data.yaml names",
        "Skipping duplicate COCO image id 1 in a.json (keeping the first)",
        "Skipping symlinked directory (symlinked directories are not descended): d",
    ],
)
def test_non_loss_warnings_are_not_rejections(message: str) -> None:
    enforce_source_integrity([message], source="fixture", strict=True)  # does not raise


def test_dataset_id_replacement_keeps_runtime_storage_options() -> None:
    # dataset_id is applied with dataclasses.replace, and datamaite holds the
    # fsspec options a remote root was opened with in an InitVar that replace()
    # resets to empty — which leaves a private-bucket dataset unable to reopen
    # its own images. Asserted on the helper because a local root legitimately
    # carries no options at all.
    source = ObjectDetectionDataset(samples=(), _storage_options={"anon": False, "key": "secret"})

    renamed = _with_dataset_id(source, "renamed")

    assert renamed.dataset_id == "renamed"
    assert renamed._runtime_storage_options == {"anon": False, "key": "secret"}


class TestToCallerAnchoredPath:
    def test_local_relative_override_becomes_a_file_uri_anchored_at_the_working_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        anchored = to_caller_anchored_path("labels")

        assert anchored == (tmp_path / "labels").resolve().as_uri()
        assert anchored.startswith("file://")

    def test_remote_url_string_is_returned_unchanged(self) -> None:
        assert to_caller_anchored_path("s3://bucket/labels") == "s3://bucket/labels"

    def test_remote_path_object_is_returned_as_the_same_object(self) -> None:
        # Stringifying would drop the options (credentials) configured on it.
        override = UPath("memory://bucket/labels", some_option="kept")

        assert to_caller_anchored_path(override) is override
