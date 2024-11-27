import json

import numpy as np
import pytest
import torch
from PIL import Image

from jatic_ri.object_detection.models import (
    InvalidInputBatchShapeError,
    InvalidModelNameError,
    MissingConfigFileError,
    TorchvisionODModel,
)


def test_integration_coco_dataset():
    """
    Integration test to validate the torchvision wrapper on a sample image from COCO dataset.

    This test runs an end-to-end check ensuring that the wrapper can successfully load a model,
    process the input data, and make predictions on real-world data.
    """
    # PIL is HWC, MAITE expects CHW
    img = Image.open("tests/testing_utilities/example_data/coco_dataset/000000037777.jpg")
    img_chw = [np.transpose(np.array(img), (2, 0, 1))]

    # somewhat arbitrary choice of model
    torchvision_wrapper = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device="cpu")
    prediction = torchvision_wrapper(input_batch=img_chw)

    # there are *many* oranges in the test image and any decent model *should* detect an orange.
    # if this test ever fails, its possible that its because we are using a poor model and hence
    # a false positive, but more likely because we have not preprocessed the model input correctly
    assert "orange" in [torchvision_wrapper.index2label[int(i)] for i in prediction[0].labels]

    # here we check that a bbox is never beyond the image boundary
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for box in prediction[0].boxes:
        x0, y0, x1, y1 = box
        if min(x0, x1) < xmin:
            xmin = min(x0, x1)
        if max(x0, x1) > xmax:
            xmax = max(x0, x1)
        if min(y0, y1) < ymin:
            ymin = min(y0, y1)
        if max(y0, y1) > ymax:
            ymax = max(y0, y1)

    assert 0 <= int(xmin) <= img.width
    assert 0 <= int(xmax) <= img.width
    assert 0 <= int(ymin) <= img.height
    assert 0 <= int(ymax) <= img.height


def test_valid_user_weights_load(tmpdir):
    """
    Test that can load user-supplied weights.

    We download a pre-trained model from torchvision, save it to disk
    as a pickle and then confirm that it can be loaded correctly.
    """

    # download
    model_wrapper = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")

    config_path = tmpdir / "config.json"
    pickle_path = tmpdir / "my_pickle.pt"

    # save metadata and state_dict to disk
    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    # reload from disk and confirm equality with original model
    model_wrapper_2 = TorchvisionODModel(
        model_name="ssdlite320_mobilenet_v3_large", weights_path=pickle_path, config_path=config_path
    )

    assert model_wrapper.index2label == model_wrapper_2.index2label

    # check that scores from random prediction from both models identical
    # maybe overkill, but useful smoke-test
    random_img = np.random.random(size=(3, 100, 100))
    assert (
        model_wrapper(input_batch=[random_img])[0].scores.numpy()
        == model_wrapper_2(input_batch=[random_img])[0].scores.numpy()
    ).all()


def test_missing_config(tmpdir):
    """Test that initializing with an invalid model name raises InvalidModelNameError."""
    model_wrapper = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
    config_path = tmpdir / "config.json"
    pickle_path = tmpdir / "my_pickle.pt"
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)
    with pytest.raises(MissingConfigFileError):
        _ = TorchvisionODModel(
            model_name="ssdlite320_mobilenet_v3_large", weights_path=pickle_path, config_path=config_path
        )


def test_valid_model_initialization():
    """Test model initialization for model name."""
    device = "cpu"
    model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device=device)
    assert model.name == "ssdlite320_mobilenet_v3_large"


def test_invalid_model_name():
    """Test that initializing with an invalid model name raises InvalidModelNameError."""
    with pytest.raises(InvalidModelNameError):
        TorchvisionODModel(model_name="invalid_model", device="cpu")


def test_call_invalid_shape():
    """Test that calling the model with an invalid shape raises InvalidInputBatchShape."""
    model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device="cpu")
    invalid_batch = torch.randn(1, 224, 224, 3)  # HWC, but we need CHW
    with pytest.raises(InvalidInputBatchShapeError):
        model(input_batch=invalid_batch)
