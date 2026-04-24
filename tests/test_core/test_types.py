"""Tests for checkmaite.core._types module."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from PIL import Image as PILImage
from pydantic import BaseModel, ValidationError

from checkmaite.core._types import DataFrame, Device, Image


class ModelWithImage(BaseModel):
    """Test model with Image field."""

    model_config = {"arbitrary_types_allowed": True}

    img: Image


class ModelWithDataFrame(BaseModel):
    """Test model with DataFrame field."""

    model_config = {"arbitrary_types_allowed": True}

    data: DataFrame


class ModelWithDevice(BaseModel):
    """Test model with Device field."""

    model_config = {"arbitrary_types_allowed": True}

    device: Device


def test_image_from_pil():
    """Test Image annotation accepts PIL Image directly."""
    pil_img = PILImage.new("RGB", (100, 100), color="red")
    model = ModelWithImage(img=pil_img)
    assert isinstance(model.img, PILImage.Image)


def test_image_from_path(tmp_path):
    """Test Image annotation converts path to PIL Image."""
    img_path = tmp_path / "test.png"
    pil_img = PILImage.new("RGB", (50, 50), color="blue")
    pil_img.save(img_path)

    model = ModelWithImage(img=str(img_path))
    assert isinstance(model.img, PILImage.Image)

    model2 = ModelWithImage(img=img_path)
    assert isinstance(model2.img, PILImage.Image)


def test_image_from_bytes(tmp_path):
    """Test Image annotation converts bytes to PIL Image."""
    img_path = tmp_path / "test.png"
    pil_img = PILImage.new("RGB", (50, 50), color="green")
    pil_img.save(img_path)

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    model = ModelWithImage(img=img_bytes)
    assert isinstance(model.img, PILImage.Image)


def test_image_from_buffered_io(tmp_path):
    """Test Image annotation converts BufferedIOBase to PIL Image."""
    img_path = tmp_path / "test.png"
    pil_img = PILImage.new("RGB", (50, 50), color="yellow")
    pil_img.save(img_path)

    with open(img_path, "rb") as f:
        model = ModelWithImage(img=f)
        assert isinstance(model.img, PILImage.Image)


def test_image_from_matplotlib_figure():
    """Test Image annotation converts matplotlib Figure to PIL Image."""
    fig = plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])

    model = ModelWithImage(img=fig)
    assert isinstance(model.img, PILImage.Image)

    plt.close(fig)


def test_dataframe_from_pandas():
    """Test DataFrame annotation accepts pandas DataFrame directly."""
    test_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    model = ModelWithDataFrame(data=test_data)
    assert isinstance(model.data, pd.DataFrame)
    assert model.data.equals(test_data)


def test_device_from_string():
    """Test Device annotation converts string to torch.device."""
    model = ModelWithDevice(device="cpu")
    assert isinstance(model.device, torch.device)
    assert model.device.type == "cpu"


def test_device_from_torch_device():
    """Test Device annotation accepts torch.device directly."""
    device = torch.device("cpu")
    model = ModelWithDevice(device=device)
    assert isinstance(model.device, torch.device)
    assert model.device == device


def test_device_from_none():
    """Test Device annotation auto-detects device from None."""
    model = ModelWithDevice(device=None)
    assert isinstance(model.device, torch.device)
    # Should be one of cpu, cuda, or mps depending on availability
    assert model.device.type in ["cpu", "cuda", "mps"]


def test_dataframe_rejects_non_pandas_input():
    """Test DataFrame annotation enforces a pandas-only boundary."""

    class FakeSparkDataFrame:
        pass

    with pytest.raises(ValidationError):
        ModelWithDataFrame(data=FakeSparkDataFrame())
