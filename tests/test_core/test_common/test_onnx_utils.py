import numpy as np
import pytest

from checkmaite.core._utils import _normalize_image, prepare_jatic_onnx_image_batch


def _metadata(*, height: int = -1, width: int = -1, batch_size: int = -1, channels: str = "RGB") -> dict:
    return {
        "interface": {"name": "JATIC_ONNX", "version": "v1"},
        "io": {
            "batchSize": batch_size,
            "interface": "IMAGE_CLASSIFICATION",
            "input": {"channels": channels, "height": height, "width": width},
            "output": {"nClasses": 3},
        },
        "index2label": {"0": "background", "1": "cat", "2": "dog"},
    }


def test_normalize_image_scales_uint8_to_unit_interval():
    image = np.array([[[0, 127, 255]]], dtype=np.uint8)

    normalized = _normalize_image(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, np.array([[[0.0, 127 / 255, 1.0]]], dtype=np.float32))


def test_normalize_image_rejects_float_images_above_unit_interval():
    image = np.array([[[0.0, 1.01]]], dtype=np.float32)

    with pytest.raises(ValueError, match="range"):
        _normalize_image(image)


def test_normalize_image_leaves_unit_interval_float_images_unchanged():
    image = np.array([[[0.0, 0.25, 1.0]]], dtype=np.float32)

    normalized = _normalize_image(image)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, image)


def test_prepare_jatic_onnx_image_batch_rejects_non_chw_input():
    hwc_image = np.zeros((10, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="CHW-ordering"):
        prepare_jatic_onnx_image_batch([hwc_image], _metadata())


def test_normalize_image_rejects_negative_integer_images():
    image = np.array([[[-1, 0, 1]]], dtype=np.int8)

    with pytest.raises(ValueError, match="non-negative"):
        _normalize_image(image)


def test_prepare_jatic_onnx_image_batch_resizes_to_target_height_and_width():
    image = np.zeros((3, 10, 20), dtype=np.uint8)
    image[:, 2:8, 5:15] = 255

    batch, original_sizes = prepare_jatic_onnx_image_batch([image], _metadata(height=5, width=7))

    assert batch.shape == (1, 3, 5, 7)
    assert batch.dtype == np.float32
    assert original_sizes == [(10, 20)]
    assert 0.0 <= float(batch.min()) <= float(batch.max()) <= 1.0


@pytest.mark.parametrize(
    "image",
    [
        np.array(
            [
                [[0, 255], [64, 128]],
                [[255, 0], [64, 128]],
                [[0, 0], [255, 128]],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [[0, 255], [64, 128]],
                [[255, 0], [64, 128]],
                [[0, 0], [255, 128]],
                [[255, 255], [255, 255]],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [[0.0, 1.0], [0.25, 0.5]],
                [[1.0, 0.0], [0.25, 0.5]],
                [[0.0, 0.0], [1.0, 0.5]],
            ],
            dtype=np.float32,
        ),
    ],
)
def test_prepare_jatic_onnx_image_batch_converts_rgb_like_inputs_to_grayscale(image):
    rgb = image[:3].astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        rgb = rgb / float(np.iinfo(image.dtype).max)
    expected = np.tensordot(np.array([0.299, 0.587, 0.114], dtype=np.float32), rgb, axes=(0, 0))[None, ...]

    batch, original_sizes = prepare_jatic_onnx_image_batch([image], _metadata(channels="GRAYSCALE"))

    assert batch.shape == (1, 1, 2, 2)
    assert batch.dtype == np.float32
    assert original_sizes == [(2, 2)]
    np.testing.assert_allclose(batch[0], expected, rtol=1e-6)


def test_prepare_jatic_onnx_image_batch_normalizes_single_channel_grayscale_input():
    image = np.array([[[0, 255], [127, 64]]], dtype=np.uint8)

    batch, original_sizes = prepare_jatic_onnx_image_batch([image], _metadata(channels="GRAYSCALE"))

    assert batch.shape == (1, 1, 2, 2)
    assert batch.dtype == np.float32
    assert original_sizes == [(2, 2)]
    np.testing.assert_allclose(batch[0], image.astype(np.float32) / 255)


def test_prepare_jatic_onnx_image_batch_rejects_negative_integer_before_grayscale_conversion():
    image = np.array(
        [
            [[0, 1]],
            [[-1, 2]],
            [[0, 3]],
        ],
        dtype=np.int16,
    )

    with pytest.raises(ValueError, match="non-negative"):
        prepare_jatic_onnx_image_batch([image], _metadata(channels="GRAYSCALE"))
