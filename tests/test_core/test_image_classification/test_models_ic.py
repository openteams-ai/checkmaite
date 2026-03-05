import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from checkmaite.core.image_classification.models import TorchvisionICModel

ROOT = Path(__file__).parents[2] / "data_for_tests"


@pytest.fixture(scope="session")
def dummy_cpu_image_batch():
    return torch.testing.make_tensor(
        (3, 3, 11, 17),
        low=0,
        high=torch.iinfo(torch.uint8).max,
        dtype=torch.uint8,
        device="cpu",
    )


@pytest.mark.parametrize("model_name", ["alexnet", "resnext50_32x4d"])
def test_torchvision_integration_dataset(model_name: str):
    """
    Integration test to validate the torchvision wrapper on a sample image from COCO dataset.

    This test runs an end-to-end check ensuring that the wrapper can successfully load a model,
    process the input data, and make predictions on real-world data.
    """
    # PIL is HWC, MAITE expects CHW
    img = Image.open(ROOT / "coco_dataset/000000087038.jpg")
    img_chw = [np.transpose(np.array(img), (2, 0, 1))]

    torchvision_wrapper = TorchvisionICModel(model_name=model_name, device="cpu")
    prediction = torchvision_wrapper(input_batch=img_chw)
    assert len(prediction) == 1
    assert prediction[0].shape == (1000,)
    assert torchvision_wrapper.name == model_name
    # It's a bicycle, but close enough, there's many other objects in the image
    assert torchvision_wrapper.index2label[prediction[0].argmax().item()] == "unicycle"


def test_torchvision_valid_user_weights_load(tmpdir, dummy_cpu_image_batch):
    # download
    model_wrapper = TorchvisionICModel(model_name="alexnet")

    config_path = tmpdir / "config.json"
    pickle_path = tmpdir / "my_pickle.pt"

    # save metadata and state_dict to disk
    with open(config_path, "w") as f:
        json.dump({"index2label": model_wrapper.index2label}, f)
    _ = torch.save(model_wrapper.model.state_dict(), pickle_path)

    # reload from disk and confirm equality with original model
    model_wrapper_2 = TorchvisionICModel(model_name="alexnet", weights_path=pickle_path, config_path=config_path)

    assert model_wrapper.index2label == model_wrapper_2.index2label

    # check that scores from random prediction from both models identical
    # maybe overkill, but useful smoke-test
    random_img = dummy_cpu_image_batch[0]
    assert (model_wrapper(input_batch=[random_img])[0] == model_wrapper_2(input_batch=[random_img])[0]).all()

    assert all(
        re.match(r"alexnet_[0-9a-f]{8}$", model_id)
        for model_id in [model_wrapper.metadata["id"], model_wrapper_2.metadata["id"]]
    )
    assert model_wrapper.metadata["id"] != model_wrapper_2.metadata["id"]


def test_torchvision_invalid_model_name():
    with pytest.raises(ValueError):
        TorchvisionICModel(model_name="invalid_model", device="cpu")


def test_torchvision_call_invalid_shape():
    model = TorchvisionICModel(model_name="alexnet", device="cpu")
    invalid_batch = torch.randn(1, 224, 224, 3)  # HWC, but we need CHW
    with pytest.raises(ValueError):
        model(input_batch=[invalid_batch])
