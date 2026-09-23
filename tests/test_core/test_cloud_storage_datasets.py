"""Boundary tests for loading datasets from non-local (fsspec) roots.

These use fsspec's in-memory filesystem: they exercise the same code path any
cloud backend takes (``s3://``, ``gs://``, ``az://``) without credentials or a
network. Remote roots are handled entirely inside datamaite 0.5.0's
storage-agnostic I/O; checkmaite's job is to not get in the way — not to
convert a URL through ``Path``, and not to drop the fsspec options a dataset
was opened with when it stamps the checkmaite-facing ``dataset_id``.
"""

import io
import shutil
from pathlib import Path

import fsspec
import numpy as np
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from PIL import Image
from upath import UPath

from checkmaite.core.image_classification.dataset_loaders import load_yolo_classification_dataset
from checkmaite.core.object_detection.dataset_loaders import (
    load_coco_detection_dataset,
    load_visdrone_detection_dataset,
    load_yolo_detection_dataset,
)

TEST_DATA_DIR = Path(__file__).parents[1] / "data_for_tests"
BUCKET = "memory://checkmaite-test"


@pytest.fixture
def memory_fs():
    fs = fsspec.filesystem("memory")
    yield fs
    if fs.exists(BUCKET):
        fs.rm(BUCKET, recursive=True)


def _upload_tree(fs, local: Path, remote: str) -> None:
    for path in sorted(local.rglob("*")):
        if path.is_file():
            fs.pipe_file(f"{remote}/{path.relative_to(local).as_posix()}", path.read_bytes())


@pytest.fixture
def memory_coco(memory_fs):
    _upload_tree(memory_fs, TEST_DATA_DIR / "coco_dataset", f"{BUCKET}/coco")
    return f"{BUCKET}/coco"


@pytest.fixture
def memory_yolo(memory_fs):
    _upload_tree(memory_fs, TEST_DATA_DIR / "yolo_dataset", f"{BUCKET}/yolo")
    return f"{BUCKET}/yolo"


@pytest.fixture
def memory_visdrone(memory_fs):
    _upload_tree(memory_fs, TEST_DATA_DIR / "visdrone_dataset", f"{BUCKET}/visdrone")
    return f"{BUCKET}/visdrone"


@pytest.fixture
def memory_classification(memory_fs):
    for split in ("train", "test"):
        for class_name in ("cat", "dog"):
            for index in range(2):
                buffer = io.BytesIO()
                Image.new("RGB", (16, 12), color=(index * 40, 0, 0)).save(buffer, format="JPEG")
                memory_fs.pipe_file(f"{BUCKET}/classification/{split}/{class_name}/{index}.jpg", buffer.getvalue())
    return f"{BUCKET}/classification"


def _assert_decodes(dataset) -> None:
    """A remote dataset must still be able to reopen its own images lazily."""
    image, _target, metadata = dataset[0]
    assert isinstance(image, np.ndarray)
    assert image.shape[0] == 3
    assert "id" in metadata


class TestRemoteObjectDetectionRoots:
    def test_coco_loads_from_remote_root(self, memory_coco) -> None:
        dataset = load_coco_detection_dataset(memory_coco, f"{memory_coco}/ann_file.json", dataset_id="memory-coco")

        assert dataset.metadata["id"] == "memory-coco"
        assert len(dataset) == 4
        assert dataset.num_detections == 57
        _assert_decodes(dataset)

    def test_yolo_loads_from_remote_root(self, memory_yolo) -> None:
        dataset = load_yolo_detection_dataset(f"{memory_yolo}/dataset.yaml", dataset_id="memory-yolo")

        assert dataset.metadata["id"] == "memory-yolo"
        assert len(dataset) == 4
        assert dataset.num_detections == 56
        _assert_decodes(dataset)

    def test_yolo_remote_ann_dir_override_is_not_reanchored(self, memory_yolo, memory_fs) -> None:
        for path in memory_fs.ls(f"{memory_yolo}/labels", detail=False):
            memory_fs.pipe_file(f"{memory_yolo}/custom_labels/{Path(path).name}", memory_fs.cat_file(path))
        memory_fs.rm(f"{memory_yolo}/labels", recursive=True)

        dataset = load_yolo_detection_dataset(f"{memory_yolo}/dataset.yaml", ann_dir=f"{memory_yolo}/custom_labels")

        assert dataset.num_detections == 56

    def test_visdrone_loads_from_remote_root(self, memory_visdrone) -> None:
        dataset = load_visdrone_detection_dataset(memory_visdrone, dataset_id="memory-visdrone")

        assert dataset.metadata["id"] == "memory-visdrone"
        assert len(dataset) == 3
        _assert_decodes(dataset)


class TestRemoteImageClassificationRoots:
    def test_yolo_classification_loads_from_remote_root(self, memory_classification) -> None:
        dataset = load_yolo_classification_dataset(memory_classification, split="test", dataset_id="memory-ic")

        assert dataset.metadata["id"] == "memory-ic"
        assert dataset.metadata["index2label"] == {0: "cat", 1: "dog"}
        assert len(dataset) == 4
        _assert_decodes(dataset)


class TestRemoteRootsAreNotTreatedAsLocalPaths:
    """A URL must survive the boundary intact.

    ``Path(str("memory://bucket/x"))`` collapses the ``//`` and produces a
    nonexistent relative path, so any local-path handling applied to a remote
    root turns into a confusing filesystem error.
    """

    def test_remote_root_is_not_collapsed_by_local_path_handling(self, memory_coco) -> None:
        assert str(Path(memory_coco)) != memory_coco  # the trap this guards

        dataset = load_coco_detection_dataset(memory_coco, f"{memory_coco}/ann_file.json")

        assert len(dataset) == 4

    def test_remote_dataset_keeps_working_after_dataset_id_is_applied(self, memory_classification) -> None:
        dataset = load_yolo_classification_dataset(memory_classification, split="test", dataset_id="renamed")

        # Images are decoded lazily, so this only works if the runtime storage
        # options survived the dataset_id replacement.
        assert all(dataset.get_input(index).shape[0] == 3 for index in range(len(dataset)))


PRIVATE = "memory://checkmaite-private"
# Not a credential: the fake backend below only checks that options arrive.
FAKE_TOKEN = "secret"  # noqa: S105


class _PrivateMemoryFileSystem(MemoryFileSystem):
    """In-memory backend that refuses access without ``token=FAKE_TOKEN``.

    Stands in for s3fs/gcsfs/adlfs credentials or endpoint options: it only
    works if the options configured on a ``UPath`` actually reach the
    filesystem datamaite opens.
    """

    protocol = "memory"

    def __init__(self, *args, token=None, **kwargs):
        self.token = token
        super().__init__(*args, **kwargs)

    def _check(self) -> None:
        if self.token != FAKE_TOKEN:
            raise PermissionError("private store requires the fake token")

    def ls(self, path, detail=True, **kwargs):
        self._check()
        return super().ls(path, detail=detail, **kwargs)

    def info(self, path, **kwargs):
        self._check()
        return super().info(path, **kwargs)

    def _open(self, path, mode="rb", block_size=None, autocommit=True, cache_options=None, **kwargs):
        self._check()
        return super()._open(path, mode, block_size, autocommit, cache_options, **kwargs)


@pytest.fixture
def private_fs():
    """Swap the authenticated backend in for ``memory://``, and always restore it."""
    fsspec.register_implementation("memory", _PrivateMemoryFileSystem, clobber=True)
    _PrivateMemoryFileSystem.clear_instance_cache()
    fs = fsspec.filesystem("memory", token=FAKE_TOKEN, skip_instance_cache=True)
    try:
        yield fs
    finally:
        if fs.exists(PRIVATE):
            fs.rm(PRIVATE, recursive=True)
        fsspec.register_implementation("memory", MemoryFileSystem, clobber=True)
        _PrivateMemoryFileSystem.clear_instance_cache()
        MemoryFileSystem.clear_instance_cache()


class TestCredentialedUPathRoots:
    """A configured ``UPath`` must reach datamaite as an object, not a string.

    ``str(UPath(..., token=...))`` keeps the URL and drops the options, so the
    backend sees an anonymous request.
    """

    def test_backend_really_enforces_credentials(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "coco_dataset", f"{PRIVATE}/coco")
        with pytest.raises((PermissionError, FileNotFoundError)):
            load_coco_detection_dataset(f"{PRIVATE}/coco", f"{PRIVATE}/coco/ann_file.json")

    def test_coco(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "coco_dataset", f"{PRIVATE}/coco")

        dataset = load_coco_detection_dataset(
            UPath(f"{PRIVATE}/coco", token=FAKE_TOKEN), f"{PRIVATE}/coco/ann_file.json"
        )

        assert len(dataset) == 4
        assert dataset.num_detections == 57
        _assert_decodes(dataset)

    def test_coco_remote_ann_file_path_object_keeps_its_options(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "coco_dataset", f"{PRIVATE}/coco")

        dataset = load_coco_detection_dataset(
            UPath(f"{PRIVATE}/coco", token=FAKE_TOKEN),
            UPath(f"{PRIVATE}/coco/ann_file.json", token=FAKE_TOKEN),
        )

        assert dataset.num_detections == 57

    def test_yolo(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "yolo_dataset", f"{PRIVATE}/yolo")

        dataset = load_yolo_detection_dataset(UPath(f"{PRIVATE}/yolo/dataset.yaml", token=FAKE_TOKEN))

        assert len(dataset) == 4
        assert dataset.num_detections == 56
        _assert_decodes(dataset)

    def test_yolo_remote_ann_dir_path_object_keeps_its_options(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "yolo_dataset", f"{PRIVATE}/yolo")

        dataset = load_yolo_detection_dataset(
            UPath(f"{PRIVATE}/yolo/dataset.yaml", token=FAKE_TOKEN),
            ann_dir=UPath(f"{PRIVATE}/yolo/labels", token=FAKE_TOKEN),
        )

        assert dataset.num_detections == 56

    def test_visdrone(self, private_fs) -> None:
        _upload_tree(private_fs, TEST_DATA_DIR / "visdrone_dataset", f"{PRIVATE}/visdrone")

        dataset = load_visdrone_detection_dataset(UPath(f"{PRIVATE}/visdrone", token=FAKE_TOKEN))

        assert len(dataset) == 3
        _assert_decodes(dataset)

    def test_yolo_classification(self, private_fs) -> None:
        for class_name in ("cat", "dog"):
            for index in range(2):
                buffer = io.BytesIO()
                Image.new("RGB", (16, 12)).save(buffer, format="JPEG")
                private_fs.pipe_file(f"{PRIVATE}/ic/test/{class_name}/{index}.jpg", buffer.getvalue())

        dataset = load_yolo_classification_dataset(UPath(f"{PRIVATE}/ic", token=FAKE_TOKEN), split="test")

        assert len(dataset) == 4
        _assert_decodes(dataset)


class TestLocalOverridesUnderRemoteRoots:
    """A local annotation override keeps its local identity under a remote root.

    Forwarded as a bare absolute path, datamaite would look it up on the remote
    root's backend: YOLO then loaded every image with no labels at all.
    """

    def test_coco_local_ann_file(self, memory_fs, tmp_path, monkeypatch) -> None:
        _upload_tree(memory_fs, TEST_DATA_DIR / "coco_dataset", f"{BUCKET}/coco")
        memory_fs.rm(f"{BUCKET}/coco/ann_file.json")  # the only copy is local
        shutil.copy(TEST_DATA_DIR / "coco_dataset" / "ann_file.json", tmp_path / "ann_file.json")
        monkeypatch.chdir(tmp_path)

        absolute = load_coco_detection_dataset(f"{BUCKET}/coco", str(tmp_path / "ann_file.json"))
        relative = load_coco_detection_dataset(f"{BUCKET}/coco", "ann_file.json")

        for dataset in (absolute, relative):
            assert len(dataset) == 4
            assert dataset.num_detections == 57
            _assert_decodes(dataset)

    def test_yolo_local_ann_dir_is_refused_rather_than_silently_unlabelled(self, memory_fs) -> None:
        # datamaite 0.5.0 cannot relate a local label path to a remote root.
        _upload_tree(memory_fs, TEST_DATA_DIR / "yolo_dataset", f"{BUCKET}/yolo")
        memory_fs.rm(f"{BUCKET}/yolo/labels", recursive=True)

        with pytest.raises(ValueError, match="local ann_dir .* not supported by datamaite 0.5.0"):
            load_yolo_detection_dataset(
                f"{BUCKET}/yolo/dataset.yaml", ann_dir=str(TEST_DATA_DIR / "yolo_dataset" / "labels")
            )
