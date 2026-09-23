# Native datamaite datasets

`CheckMAITE` delegates still-image dataset loading to
[datamaite](https://pypi.org/project/datamaite/) and returns datamaite's native
MAITE-compatible dataset objects directly. CheckMAITE does not maintain a
second set of COCO, YOLO, or VisDrone dataset classes.

## Image classification

```python
from checkmaite.core.image_classification.dataset_loaders import (
    load_yolo_classification_dataset,
)

dataset = load_yolo_classification_dataset(
    "/path/to/yolo-classification",
    split="test",
    dataset_id="evaluation",
)
image, one_hot_target, metadata = dataset[0]
```

## Object detection

```python
from checkmaite.core.object_detection.dataset_loaders import (
    load_coco_detection_dataset,
)

dataset = load_coco_detection_dataset(
    root="/path/to/coco/images",
    ann_file="/path/to/coco/instances.json",
    dataset_id="evaluation",
)
image, target, metadata = dataset[0]
```

The returned image and target fields are NumPy arrays, which satisfy MAITE's
`ArrayLike` contract. Torch-specific models should convert their input batch at
the model boundary rather than requiring every dataset to store Torch tensors.

datamaite 0.5.0 provides MAITE fieldwise access
(`get_input`/`get_target`/`get_metadata`), full COCO `images[]` fields and
per-box VisDrone truncation/occlusion values in datum metadata, native YOLO
`split`/`yaml_file`/`ann_dir` loader options, recursive image discovery below
YOLO classification class directories, and storage-agnostic I/O — so the
loaders here are thin factories with no checkmaite-side adapters.

## Remote dataset roots

Any root may be a cloud URL (`s3://`, `gs://`, `az://`) as well as a local path.
Install the matching provider extra — `checkmaite[aws]`, `checkmaite[gcs]`,
`checkmaite[azure]`, or `checkmaite[cloud]` for all three — and pass fsspec
credentials through `storage_options`:

```python
dataset = load_coco_detection_dataset(
    root="s3://bucket/coco",
    ann_file="s3://bucket/coco/instances.json",
    storage_options={"anon": False},
)
```

Images are decoded lazily, so a remote dataset keeps its storage options for the
lifetime of the object.

## Relative annotation paths

`ann_file` and `ann_dir` are resolved against the **working directory**, which is
what CheckMAITE's loaders have always done. datamaite anchors a relative override
to the dataset root instead, so CheckMAITE makes local relative overrides
absolute before forwarding them, as `file://` URIs so they stay local even under
a remote dataset root. Remote URIs are passed through untouched, and a configured
`UPath` (for example one carrying credentials) is passed through as an object so
its filesystem options are kept.

One combination isn't supported yet: datamaite 0.5.0 can't relate a local YOLO
`ann_dir` to a remote dataset root, so that combination raises `ValueError`
instead of loading images with no labels. A local COCO `ann_file` under a remote
root works.

## Malformed annotations

datamaite's loaders are best-effort: a row they cannot parse is dropped with a
warning. By default CheckMAITE refuses such a dataset instead, because loading a
rejected annotation as background silently biases every detection metric computed
from it. The error aggregates datamaite's own `file:line` diagnostics:

```text
checkmaite.core._common.dataset_utils.DatasetSourceError: 2 source record(s) were
rejected while loading YOLO dataset /data/set/dataset.yaml (split 'train'). ...
  - Skipping malformed YOLO label row /data/set/labels/000000037777.txt:1 (expected 5 or 6 fields)
  - Skipping YOLO label row /data/set/labels/000000037777.txt:3 with out-of-range normalized bbox
```

Pass `strict_annotations=False` to load the parseable records anyway; the same
diagnostic is then emitted as a warning. An image with no label file at all is a
legitimate background image and is never an error.

This check is **best effort**. `strict_annotations=True` promotes the datamaite
row-rejection warnings CheckMAITE recognizes to `DatasetSourceError`. It works by
matching datamaite's log messages, so a loss datamaite reports in a phrasing
CheckMAITE doesn't recognize, or doesn't report at all, still gets through. It
does not currently guarantee complete source integrity.
