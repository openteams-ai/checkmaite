import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from checkmaite.core.image_classification.models import OnnxICModel, TorchvisionICModel
from checkmaite.core.image_classification.models import load_models as load_ic_models

ROOT = Path(__file__).parents[2] / "data_for_tests"
HAS_ONNX_DEPS = importlib.util.find_spec("onnx") is not None and importlib.util.find_spec("onnxruntime") is not None


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


def _save_constant_onnx_model(
    path: Path,
    *,
    input_shape: list[int],
    outputs: dict[str, np.ndarray],
) -> None:
    """Create a tiny ONNX model that ignores its image input and returns fixed tensors.

    The generated graph has the JATIC_ONNX-required input name, ``image``, but it does not perform real inference.
    Each requested output is produced by an ONNX ``Constant`` node. This makes the image-classification ONNX wrapper
    tests deterministic and fast: they exercise model loading, metadata validation, ONNX Runtime invocation, and output
    conversion without depending on a large trained network or internet downloads.
    """
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, input_shape)
    output_infos = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)) for name, value in outputs.items()
    ]
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[name],
            value=helper.make_tensor(
                name=f"{name}_value",
                data_type=TensorProto.FLOAT,
                dims=list(value.shape),
                vals=value.astype(np.float32).ravel().tolist(),
            ),
        )
        for name, value in outputs.items()
    ]
    graph = helper.make_graph(nodes, "constant_jatic_onnx_test_model", [input_info], output_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


def _write_onnx_metadata(path: Path, *, io_interface: str, batch_size: int, output: dict[str, int]) -> None:
    path.write_text(
        json.dumps(
            {
                "interface": {"name": "JATIC_ONNX", "version": "v1"},
                "io": {
                    "batchSize": batch_size,
                    "interface": io_interface,
                    "input": {"channels": "RGB", "height": -1, "width": -1},
                    "output": output,
                },
                "index2label": {"0": "background", "1": "cat", "2": "dog"},
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.skipif(not HAS_ONNX_DEPS, reason="ONNX wrapper tests require the optional ONNX dependencies.")
def test_onnx_ic_model_returns_jatic_scores(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    expected_scores = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]], dtype=np.float32)
    _save_constant_onnx_model(model_path, input_shape=[2, 3, 4, 5], outputs={"scores": expected_scores})
    _write_onnx_metadata(config_path, io_interface="IMAGE_CLASSIFICATION", batch_size=2, output={"nClasses": 3})

    model = OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")
    predictions = model([np.zeros((3, 4, 5), dtype=np.uint8), np.ones((3, 4, 5), dtype=np.uint8)])

    assert model.name == "jatic_onnx"
    assert model.index2label == {0: "background", 1: "cat", 2: "dog"}
    assert len(predictions) == 2
    assert torch.equal(predictions[0], torch.as_tensor(expected_scores[0]))
    assert torch.equal(predictions[1], torch.as_tensor(expected_scores[1]))


@pytest.mark.skipif(not HAS_ONNX_DEPS, reason="ONNX wrapper tests require the optional ONNX dependencies.")
def test_onnx_ic_load_models_dispatch(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path,
        input_shape=[1, 3, 4, 5],
        outputs={"scores": np.array([[0.1, 0.7, 0.2]], dtype=np.float32)},
    )
    _write_onnx_metadata(config_path, io_interface="IMAGE_CLASSIFICATION", batch_size=1, output={"nClasses": 3})

    loaded = load_ic_models(
        {
            "onnx_model": {
                "model_type": "jatic_onnx",
                "model_weights_path": model_path,
                "model_config_path": config_path,
            }
        },
        device="cpu",
    )

    assert isinstance(loaded["onnx_model"], OnnxICModel)


@pytest.mark.skipif(not HAS_ONNX_DEPS, reason="ONNX wrapper tests require the optional ONNX dependencies.")
def test_onnx_ic_model_requires_index2label(tmp_path: Path):
    model_path = tmp_path / "ic.onnx"
    config_path = tmp_path / "model-metadata.json"
    _save_constant_onnx_model(
        model_path,
        input_shape=[1, 3, 4, 5],
        outputs={"scores": np.array([[1.0]], dtype=np.float32)},
    )
    config_path.write_text(
        json.dumps(
            {
                "interface": {"name": "JATIC_ONNX", "version": "v1"},
                "io": {
                    "batchSize": 1,
                    "interface": "IMAGE_CLASSIFICATION",
                    "input": {"channels": "RGB", "height": 4, "width": 5},
                    "output": {"nClasses": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="index2label"):
        OnnxICModel(weights_path=model_path, config_path=config_path, device="cpu")


def test_torchvision_invalid_model_name():
    with pytest.raises(ValueError):
        TorchvisionICModel(model_name="invalid_model", device="cpu")


def test_torchvision_call_invalid_shape():
    model = TorchvisionICModel(model_name="alexnet", device="cpu")
    invalid_batch = torch.randn(1, 224, 224, 3)  # HWC, but we need CHW
    with pytest.raises(ValueError):
        model(input_batch=[invalid_batch])
