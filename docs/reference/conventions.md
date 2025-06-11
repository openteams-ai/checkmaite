# Reference Implementation Conventions

## Overview

The MAITE protocols for metrics, models, and datasets allow for a great deal of flexibility, and only define specific functionality where needed by the integrations that are nearly universal to the workflow of a model evaluation.

During the course of testing capability development, and particularly when new testing capabilities are developed or acquired as part of the JATIC program, we will likely identify integrations where the protocols lack the specificity required to smoothly incorporate those new capabilities into our existing test cases (or new ones).

This document serves to declare the specific narrowing of scope that the reference implementation (RI) will use for creating classes and objects that align to MAITE protocol compliant objects.

**Note**: These conventions ensure that the objects comply with the protocols, and any additional specificity required by these protocols must be consistent with the generalized form of the protocols.

As these conventions are incorporated into the reference implementation, they may be absorbed into the MAITE protocols in future releases. If this occurs, the reference implementation team will update the RI to align with the new protocols and remove any redundant sections from this document.

## Models

### Model conventions

- Model classes and wrappers created within the reference implementation (and any models that will be evaluated using the reference implementation) will be built with the underlying model API (pytorch, tensorflow, keras, etc.) object accessible under an attribute called `model`. For example, a thinly wrapped `fasterrcnn_resnet50_fpn` model called `wrapped` will have the underlying `fasterrcnn_resnet50_fpn` model accessible via a call to `wrapped.model`. Because `model` is not an attribute of the protocols for models, using it in a test stage or elsewhere will trigger a failure of the type checker, so use `# pyright: ignore[reportUnknownMemberType]` on those lines that depend on the `model` attribute.

- Model classes within the reference implementation will have a mapping from the ids to the names of the classes that they predict, under the attribute `index2label`. This class attribute will be of type `dict[int, str]`. Because `index2label` is not an attribute of the protocols for models, using it in a test stage or elsewhere will trigger a failure of the type checker, so use `# pyright: ignore[reportTypedDictNotRequiredAccess]` on those lines that depend on the `index2label` attribute.

- Object detection models will return MAITE-compliant object detection [targets](https://gitlab.jatic.net/jatic/cdao/maite/-/blob/main/src/maite/_internals/protocols/object_detection.py?ref_type=heads#L27), which consist of `ArrayLike` of predicted classes, `ArrayLike` of scores, and `ArrayLike` of bounding boxes.

- Image classification models will return MAITE-compliant image classification [targets](https://gitlab.jatic.net/jatic/cdao/maite/-/blob/main/src/maite/_internals/protocols/image_classification.py?ref_type=heads#L25), which consist of an `ArrayLike` of pseudo-probabilities for each class it is trained to predict.
    
- Models which find no prediction on a given target will return an `ObjectDetectionTarget` with empty `ArrayLike` elements for bounding boxes/labels.

### Supported models

In order for users to be able to bring their own models to the reference implementation, without having to write any additional code to support the loading and wrapping of that model into MAITE-compliant objects, we must provide some generalized wrappers for commonly used models. This list contains the set of models that the reference implementation has (or will) support. As development continues, this list will be updated.

#### Object Detection
- [fasterrcnn_resnet50_fpn](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- [fasterrcnn_resnet50_fpn_v2](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
- [maskrcnn_resnet50_fpn_v2](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html)
- [maskrcnn_resnet50_fpn](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html)
- [retinanet_resnet50_fpn](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html)
- [retinanet_resnet50_fpn_v2](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html)

#### Image Classification
- [alexnet](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html)
- [resnext50_32x4d](https://pytorch.org/vision/master/models/generated/torchvision.models.resnext50_32x4d.html#torchvision.models.resnext50_32x4d)

## Datasets

### Dataset conventions

- Dataset classes and wrappers created within the reference implementation (and any datasets used within the RI) will have the mapping from id to classes accessible under the attribute `index2label`, and will return a `dict[int, str]`. Because `index2label` is not an attribute of the protocols for datasets, using it in a test stage or elsewhere will trigger a failure of the type checker, so use `# pyright: ignore[reportTypedDictNotRequiredAccess]` on those lines that depend on the `index2label` attribute.

- Object detection bounding boxes will be defined as `ArrayLikes` of integers in the `xyxy` format (the top-left and bottom-right corners of the bounding box).

- Labels will be defined as `ArrayLikes` of integers, whose values map to the keys in `index2label`.

- Other datum-level metadata will be stored as strings in `DatumMetadata`.

### Supported dataset annotation formats

In order for users to be able to bring their own datasets to the reference implementation, without having to write any additional code to support the loading and wrapping of that dataset into MAITE-compliant objects, we must provide some generalized wrappers for commonly used annotation formats for loading data. This list contains the set of dataset annotation and storage formats that the reference implementation has (or will) support. This includes the ability to load the datasets from a user-provided location.

* Object detection
    * [COCO](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)
    * [YOLO](https://docs.ultralytics.com/datasets/detect/)
* Image Classification
    * [YOLO](https://docs.ultralytics.com/datasets/classify/#dataset-structure-for-yolo-classification-tasks)

## Metrics

### Metric conventions

- Classes for computing metrics within the reference implementation will follow these conventions:
  - The `Metric` class will have an attribute called `return_key` which describes the top-level performance metric:
    - E.g., for a metric that computes `map_50`, the `return_key` will be `map_50`.
    - Because `return_key` is not an attribute of the protocols for metrics, using it in a test stage or elsewhere will trigger a failure of the type checker, so use `# pyright: ignore[reportUnknownMemberType]` on those lines that depend on the `return_key` attribute.
  
  - The list of keys returned by `compute()` will include a key that matches the `return_key` of the metric, which describes the top-level performance of predictions against the ground truth provided in the `update()` method.
  
  - The values of each element in the dictionary returned by `compute()` will adhere to the `numpy.ArrayLike` type, and will consist of an array containing a single floating point value such that:
    - The value can safely be cast to a float with `float(<value>)`.
    - The value will possess the `<value>.numpy()` method.

- Datum-level metrics will not be guaranteed by the metric classes at this time, and tools that rely on datum-level metrics will need to compute them within their test stage classes by iterating through the dataset and computing the metric on a "Dataset" with only a single datum in it.
