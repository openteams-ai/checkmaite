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
from checkmaite.core.multiobject_tracking import load_mot_dataset

dataset = load_mot_dataset("path/to/videos", "flat_mp4")

stream, target, metadata = dataset[0]  # one datum per video
for frame in stream:
    frame.pixels  # uint8 array, shape (3, height, width)
```

`load_mot_dataset` calls DataMAITE's `load_mot` and always emits every frame of
every video. It also gives the dataset an id derived from its root, format and
options: CheckMAITE caches predictions and evaluations by dataset id, and
`load_mot` gives every dataset the same default id, so load through
`load_mot_dataset` or pass your own `dataset_id`.

## Run a model over the videos

The result is a MAITE MOT dataset, so it goes anywhere CheckMAITE takes a MOT
dataset. Today that is the cached prediction task, which runs a MAITE
multi-object-tracking model over every video:

```python
from modelmaite import ByteTrackMOTModel, TorchvisionODModel

from checkmaite import cached_tasks

detector = TorchvisionODModel(model_name="fasterrcnn_resnet50_fpn")
tracker = ByteTrackMOTModel(detector=detector)

predictions, _ = cached_tasks.predict(model=tracker, dataset=dataset)
```

`tracker` can be any MAITE MOT model: it takes a batch of video streams and
returns one target per video, with one set of tracks per frame. This example
uses modelmaite's ByteTrack wrapper around a torchvision detector, which needs
`modelmaite[mot]`. CheckMAITE does not install that extra.

## Evaluate a tracker

`checkmaite.core.multiobject_tracking.MaiteEvaluation` scores a MOT model on an
annotated dataset, the same way the object-detection and image-classification
`MaiteEvaluation` capabilities do. The built-in metric, computed with
[py-motmetrics](https://github.com/cheind/py-motmetrics), reports MOTA (the
headline value), MOTP, IDF1 and the number of identity switches. MOTP follows
the motmetrics convention: the mean IoU *distance* (1 - IoU) of matched boxes,
so lower is better. A score that is undefined for the evaluation is left out of
the results, such as MOTP when the tracker matched nothing; if MOTA is undefined
because the dataset has no ground-truth objects, the evaluation fails with an
error.

The metric matches boxes regardless of class, so keep only the classes you
want to score in both the dataset (`classes=`) and the tracker's output. MOT17
labels only pedestrians, while the torchvision detector above predicts every
COCO class, so this example keeps only its `person` boxes before tracking;
otherwise every other box counts as a false positive. Give the wrapped detector
its own `id`, because CheckMAITE caches results by model id.

```python
import numpy as np
from modelmaite import ByteTrackMOTModel
from modelmaite.object_detection import DetectionTarget

from checkmaite.core.multiobject_tracking import MaiteEvaluation, load_mot_dataset
from checkmaite.core.multiobject_tracking.metrics import clear_mot_metrics_factory


class PersonDetector:
    """Keep only the detector's `person` boxes."""

    def __init__(self, detector):
        self.detector = detector
        labels = detector.metadata["index2label"]
        self.person = next(i for i, name in labels.items() if name == "person")
        self.metadata = {**detector.metadata, "id": f"{detector.metadata['id']}_person"}

    def __call__(self, batch):
        targets = []
        for prediction in self.detector(batch):
            keep = np.asarray(prediction.labels) == self.person
            targets.append(
                DetectionTarget(
                    boxes=np.asarray(prediction.boxes)[keep],
                    labels=np.asarray(prediction.labels)[keep],
                    scores=np.asarray(prediction.scores)[keep],
                )
            )
        return targets


person_tracker = ByteTrackMOTModel(detector=PersonDetector(detector))
dataset = load_mot_dataset("path/to/MOT17", "motchallenge", classes={1})

run = MaiteEvaluation().run(models=[person_tracker], datasets=[dataset], metrics=[clear_mot_metrics_factory()])
run.outputs.metrics
```

Predictions with track ID `-1`, which MAITE reserves for untracked detections,
are not scored. MOTChallenge distractor classes and ignored regions are not
handled, so scores on MOTChallenge data differ from the official leaderboard.

## Supported formats

The evaluation scores every frame, and a frame without boxes counts as empty, so
detections on it are false positives. That is only correct when every frame is
labeled, so `load_mot_dataset` accepts `"motchallenge"` and `"flat_mp4"`, and
refuses a sequence whose exact frame count is unknown.

DataMAITE also reads `"tao"`, `"hmie"` and `"visdrone_video"`, but
`load_mot_dataset` refuses them for now. TAO and HMIE usually label only some
frames, which needs scoring on the labeled frames while tracking over every
frame. DataMAITE currently drops VisDrone ground-truth tracks whose id is 0.

For video classification datasets in the Hugging Face VideoFolder layout, use
`datamaite.load_vc`. See the DataMAITE documentation for each format's layout.
