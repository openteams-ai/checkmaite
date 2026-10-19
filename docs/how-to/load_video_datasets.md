# Load a video dataset

CheckMAITE reads video datasets through
[DataMAITE](https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite),
the same library it uses for image datasets. DataMAITE returns a MAITE
multi-object-tracking (MOT) dataset, which CheckMAITE accepts like any other
MAITE dataset. DataMAITE's video support, including
[PyAV](https://pyav.basswood-io.com/) for decoding MP4, is installed with
CheckMAITE.

## Load a flat folder of MP4 videos

Put the `.mp4` files directly in one folder. DataMAITE accepts H.264 and MPEG-2
video, which is what the JATIC interoperability standard `IR-3.3-S-1` requires,
and skips other codecs with a warning.

```python
from datamaite import load_mot

dataset = load_mot("path/to/videos", dataset_format="flat_mp4")
dataset = dataset.with_mot_options(empty_frame_policy="all")

stream, target, metadata = dataset[0]  # one datum per video
for frame in stream:
    frame.pixels  # uint8 array, shape (3, height, width)
```

A flat folder has no annotations, so by default no frames are emitted. Pass
`empty_frame_policy="all"` to get every frame.

## Run a model over the videos

The result is a MAITE MOT dataset, so it goes anywhere CheckMAITE takes a MOT
dataset. Today that is the cached prediction task, which runs a MAITE
multi-object-tracking model over every video:

```python
from checkmaite import cached_tasks

predictions, _ = cached_tasks.predict(model=tracker, dataset=dataset)
```

`tracker` is any MAITE MOT model: it takes a batch of video streams and returns
one target per video, with one set of tracks per frame. No built-in CheckMAITE
capability consumes MOT datasets yet.

## Other video formats

`load_mot` also reads annotated tracking datasets through `dataset_format`:
`"motchallenge"`, `"tao"`, `"visdrone_video"` and `"hmie"`. For video
classification datasets in the Hugging Face VideoFolder layout, use
`datamaite.load_vc`. See the DataMAITE documentation for each format's layout.
