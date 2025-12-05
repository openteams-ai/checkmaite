import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from jatic_ri.core.object_detection.models import TorchvisionODModel, VisdroneODModel


@pytest.fixture(scope="session")
def dummy_cpu_image_batch():
    return torch.testing.make_tensor(
        (3, 3, 11, 17),
        low=0,
        high=torch.iinfo(torch.uint8).max,
        dtype=torch.uint8,
        device="cpu",
    )


@pytest.fixture
def fake_model_location(tmp_path):
    model_name = "fake_model"
    fake_file_path = tmp_path / f"{model_name}.pth"
    fake_file_path.write_bytes(b"dummy content")

    return tmp_path, model_name


def test_visdrone_invalid_arch_name():
    with pytest.raises(ValueError):
        VisdroneODModel(arch="invalid_arch", device="cpu")


def test_visdrone_valid_model_initialization(fake_model_location):
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    assert model.name == f"visdrone-centernet-{fname}"
    assert model.metadata["id"] == "visdrone_resnet18_kitware"


def test_visdrone_metadata_fields(fake_model_location):
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    metadata = model.metadata
    assert "id" in metadata
    assert "index2label" in metadata
    assert isinstance(metadata["index2label"], dict)


def test_visdrone_call_invalid_shape(fake_model_location):
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    invalid_batch = torch.randn(224, 224, 3)  # HWC instead of CHW
    with pytest.raises(ValueError):
        model(input_batch=[invalid_batch])


def test_torchvision_integration_coco_dataset():
    """
    Integration test to validate the torchvision wrapper on a sample image from COCO dataset.

    This test runs an end-to-end check ensuring that the wrapper can successfully load a model,
    process the input data, and make predictions on real-world data.
    """
    root = Path(__file__).parents[2] / "data_for_tests"
    img = Image.open(root / "coco_dataset/000000037777.jpg")
    # PIL is HWC, MAITE expects CHW
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


def test_torchvision_valid_user_weights_load(tmpdir, dummy_cpu_image_batch):
    """
    Test that can load user-supplied weights.

    We download a pre-trained model from torchvision, save it to disk
    as a pickle and then confirm that it can be loaded correctly.
    """
    model_wrapper = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")

    config_path = tmpdir / "config.json"
    pickle_path = tmpdir / "my_pickle.pt"

    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    model_wrapper_2 = TorchvisionODModel(
        model_name="ssdlite320_mobilenet_v3_large", weights_path=pickle_path, config_path=config_path
    )

    assert model_wrapper.index2label == model_wrapper_2.index2label

    # check that scores from random prediction from both models identical
    # maybe overkill, but useful smoke-test
    random_img = dummy_cpu_image_batch[0]
    assert (
        model_wrapper(input_batch=[random_img])[0].scores.numpy()
        == model_wrapper_2(input_batch=[random_img])[0].scores.numpy()
    ).all()

    assert all(
        re.match(r"ssdlite320_mobilenet_v3_large_[0-9a-f]{8}$", model_id)
        for model_id in [model_wrapper.metadata["id"], model_wrapper_2.metadata["id"]]
    )
    assert model_wrapper.metadata["id"] != model_wrapper_2.metadata["id"]


def test_torchvision_missing_config(tmpdir):
    model_wrapper = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large")
    config_path = tmpdir / "config.json"
    pickle_path = tmpdir / "my_pickle.pt"
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)
    with pytest.raises(FileNotFoundError):
        _ = TorchvisionODModel(
            model_name="ssdlite320_mobilenet_v3_large", weights_path=pickle_path, config_path=config_path
        )


def test_torchvision_valid_model_initialization():
    device = "cpu"
    model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device=device)
    assert model.name == "ssdlite320_mobilenet_v3_large"


def test_torchvision_invalid_model_name():
    with pytest.raises(ValueError):
        TorchvisionODModel(model_name="invalid_model", device="cpu")


def test_torchvision_call_invalid_shape():
    model = TorchvisionODModel(model_name="ssdlite320_mobilenet_v3_large", device="cpu")
    invalid_batch = torch.randn(1, 224, 224, 3)  # HWC, but we need CHW
    with pytest.raises(ValueError):
        model(input_batch=invalid_batch)


@pytest.mark.real_data
def test_integration_visdrone_dataset(tmp_path):
    """
    Integration test for Visdrone model on a sample image.

    This ensures the Visdrone wrapper loads the model, processes input correctly,
    and generates valid predictions.
    """
    root = Path(__file__).parents[2] / "data_for_tests"
    img = Image.open(root / "visdrone_dataset/images/0000001_02999_d_0000005.jpg")
    img_chw = [np.transpose(np.array(img), (2, 0, 1))]

    visdrone_wrapper = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=tmp_path)
    prediction = visdrone_wrapper(input_batch=img_chw)

    # Verify at least one detection is made
    assert len(prediction) > 0
    pred = np.array(prediction[0].boxes)
    assert len(pred) > 0

    # Verify bounding boxes are within image bounds
    xmin, xmax = 0, 0
    ymin, ymax = 0, 0
    for box in pred:
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
