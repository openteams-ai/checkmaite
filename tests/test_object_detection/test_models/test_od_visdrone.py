import numpy as np
import pytest
import torch
from PIL import Image

from jatic_ri.object_detection.models import VisdroneODModel


@pytest.fixture
def fake_model_location(tmp_path):
    # model name should not include suffix
    model_name = "fake_model"
    fake_file_path = tmp_path / f"{model_name}.pth"
    fake_file_path.write_bytes(b"dummy content")

    return tmp_path, model_name


@pytest.mark.real_data
def test_integration_visdrone_dataset():
    """
    Integration test for Visdrone model on a sample image.

    This ensures the Visdrone wrapper loads the model, processes input correctly,
    and generates valid predictions.
    """
    img = Image.open("tests/testing_utilities/example_data/visdrone_dataset/images/0000001_02999_d_0000005.jpg")
    img_chw = [np.transpose(np.array(img), (2, 0, 1))]

    visdrone_wrapper = VisdroneODModel(arch="resnet18", device="cpu")
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


def test_invalid_arch_name():
    """Test that initializing with an invalid architecture raises ValueError."""
    with pytest.raises(ValueError):
        VisdroneODModel(arch="invalid_arch", device="cpu")


def test_valid_model_initialization(fake_model_location):
    """Test Visdrone model initializes correctly."""
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    assert model.name == f"visdrone-centernet-{fname}"


def test_metadata_fields(fake_model_location):
    """Test that the model metadata contains expected fields."""
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    metadata = model.metadata
    assert "id" in metadata
    assert "index2label" in metadata
    assert isinstance(metadata["index2label"], dict)


def test_call_invalid_shape(fake_model_location):
    """
    Test that calling the Visdrone model with an invalid input shape raises an error.

    The Visdrone model expects input in (C, H, W) format per image.
    """
    dir_, fname = fake_model_location
    model = VisdroneODModel(arch="resnet18", device="cpu", model_pickle_dir=dir_, model_name=fname)
    invalid_batch = torch.randn(224, 224, 3)  # HWC instead of CHW
    with pytest.raises(ValueError):
        model(input_batch=[invalid_batch])
