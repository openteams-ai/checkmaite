"""Tests for checkmaite.core.serving.rayserve module."""

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from checkmaite.core.object_detection.dataset_loaders import DetectionTarget
from checkmaite.core.serving.rayserve._clients import (
    RayImageClassificationClient,
    RayObjectDetectionClient,
    _get_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDetectionTarget:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


class _FakeODModel:
    """Minimal fake OD model for deployment tests."""

    def __init__(self, model_id="test-od"):
        self.metadata = {"id": model_id, "index2label": {0: "cat", 1: "dog"}}

    def __call__(self, input_batch):
        return [
            _FakeDetectionTarget(
                boxes=np.array([[10, 20, 30, 40]], dtype=np.float32),
                labels=np.array([0], dtype=np.int32),
                scores=np.array([0.9], dtype=np.float32),
            )
            for _ in input_batch
        ]


class _FakeICModel:
    """Minimal fake IC model for deployment tests."""

    def __init__(self, model_id="test-ic"):
        self.metadata = {"id": model_id, "index2label": {0: "cat", 1: "dog"}}

    def __call__(self, input_batch):
        return [np.array([0.8, 0.2], dtype=np.float32) for _ in input_batch]


class _FakeResponse:
    """Mock for Ray DeploymentResponse."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeMethodCaller:
    """Mock for handle.method_name.remote() pattern."""

    def __init__(self, value):
        self._value = value

    def remote(self, *args, **kwargs):
        return _FakeResponse(self._value)


class _FakeHandle:
    """Mock for DeploymentHandle with get_metadata and predict."""

    def __init__(self, metadata, predict_fn):
        self.get_metadata = _FakeMethodCaller(metadata)
        self._predict_fn = predict_fn

    @property
    def predict(self):
        return _FakePredictCaller(self._predict_fn)


class _FakePredictCaller:
    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def remote(self, *args, **kwargs):
        return _FakeResponse(self._predict_fn(*args, **kwargs))


# ---------------------------------------------------------------------------
# _get_result tests
# ---------------------------------------------------------------------------


class TestGetResult:
    def test_sync_context(self):
        """_get_result calls .result() directly when no event loop is running."""
        response = _FakeResponse(42)
        assert _get_result(response) == 42

    def test_async_context(self):
        """_get_result uses thread pool when inside an event loop."""
        response = _FakeResponse("hello")

        async def _run():
            return _get_result(response)

        result = asyncio.run(_run())
        assert result == "hello"


# ---------------------------------------------------------------------------
# RayObjectDetectionDeployment tests
# ---------------------------------------------------------------------------


class TestRayObjectDetectionDeployment:
    @pytest.fixture
    def deployment(self):
        from checkmaite.core.serving.rayserve._deployments import RayObjectDetectionDeployment

        # Access the original class behind @serve.deployment decorator
        cls = RayObjectDetectionDeployment.func_or_class
        return cls(_FakeODModel, {"model_id": "deploy-od"})

    def test_init_creates_model(self, deployment):
        assert deployment.model.metadata["id"] == "deploy-od"

    def test_get_metadata(self, deployment):
        result = asyncio.run(deployment.get_metadata())
        assert result["id"] == "deploy-od"
        assert "index2label" in result
        assert result["model_cls"] == "_FakeODModel"

    def test_predict_returns_numpy_dicts(self, deployment):
        batch = [np.random.default_rng(42).integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(2)]
        results = asyncio.run(deployment.predict(batch))

        assert len(results) == 2
        for pred in results:
            assert isinstance(pred, dict)
            assert set(pred.keys()) == {"boxes", "labels", "scores"}
            assert isinstance(pred["boxes"], np.ndarray)
            assert isinstance(pred["labels"], np.ndarray)
            assert isinstance(pred["scores"], np.ndarray)

    def test_predict_empty_batch(self, deployment):
        results = asyncio.run(deployment.predict([]))
        assert results == []


# ---------------------------------------------------------------------------
# RayImageClassificationDeployment tests
# ---------------------------------------------------------------------------


class TestRayImageClassificationDeployment:
    @pytest.fixture
    def deployment(self):
        from checkmaite.core.serving.rayserve._deployments import RayImageClassificationDeployment

        # Access the original class behind @serve.deployment decorator
        cls = RayImageClassificationDeployment.func_or_class
        return cls(_FakeICModel, {"model_id": "deploy-ic"})

    def test_init_creates_model(self, deployment):
        assert deployment.model.metadata["id"] == "deploy-ic"

    def test_get_metadata(self, deployment):
        result = asyncio.run(deployment.get_metadata())
        assert result["id"] == "deploy-ic"
        assert result["model_cls"] == "_FakeICModel"

    def test_predict_returns_numpy_arrays(self, deployment):
        batch = [np.random.default_rng(42).integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(3)]
        results = asyncio.run(deployment.predict(batch))

        assert len(results) == 3
        for pred in results:
            assert isinstance(pred, np.ndarray)

    def test_predict_empty_batch(self, deployment):
        results = asyncio.run(deployment.predict([]))
        assert results == []


# ---------------------------------------------------------------------------
# RayObjectDetectionClient tests
# ---------------------------------------------------------------------------


class TestRayObjectDetectionClient:
    @pytest.fixture
    def client(self):
        metadata = {"id": "test-od-model", "index2label": {0: "cat", 1: "dog"}}

        def predict_fn(np_batch):
            return [
                {
                    "boxes": np.array([[10, 20, 30, 40]], dtype=np.float32),
                    "labels": np.array([0], dtype=np.int32),
                    "scores": np.array([0.95], dtype=np.float32),
                }
                for _ in np_batch
            ]

        handle = _FakeHandle(metadata, predict_fn)
        return RayObjectDetectionClient(handle)

    def test_metadata_fetched_at_init(self, client):
        assert client.metadata["id"] == "test-od-model"
        assert client.metadata["index2label"] == {0: "cat", 1: "dog"}

    def test_name_property(self, client):
        assert client.name == "remote-od-test-od-model"

    def test_call_returns_detection_targets(self, client):
        images = [np.random.default_rng(42).integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(2)]
        results = client(images)

        assert len(results) == 2
        for target in results:
            assert isinstance(target, DetectionTarget)
            assert isinstance(target.boxes, np.ndarray)
            assert isinstance(target.labels, np.ndarray)
            assert isinstance(target.scores, np.ndarray)
            np.testing.assert_array_equal(target.boxes, [[10, 20, 30, 40]])
            np.testing.assert_array_equal(target.labels, [0])
            np.testing.assert_array_almost_equal(target.scores, [0.95])

    def test_call_converts_input_to_numpy(self, client):
        """Ensure torch tensors or other array-likes are converted to numpy before sending."""
        import torch

        images = [torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)]
        results = client(images)
        assert len(results) == 1
        assert isinstance(results[0], DetectionTarget)

    def test_call_empty_batch(self, client):
        results = client([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# RayImageClassificationClient tests
# ---------------------------------------------------------------------------


class TestRayImageClassificationClient:
    @pytest.fixture
    def client(self):
        metadata = {"id": "test-ic-model", "index2label": {0: "cat", 1: "dog"}}

        def predict_fn(np_batch):
            return [np.array([0.7, 0.3], dtype=np.float32) for _ in np_batch]

        handle = _FakeHandle(metadata, predict_fn)
        return RayImageClassificationClient(handle)

    def test_metadata_fetched_at_init(self, client):
        assert client.metadata["id"] == "test-ic-model"

    def test_name_property(self, client):
        assert client.name == "remote-ic-test-ic-model"

    def test_call_returns_numpy_arrays(self, client):
        images = [np.random.default_rng(42).integers(0, 255, (3, 32, 32), dtype=np.uint8) for _ in range(3)]
        results = client(images)

        assert len(results) == 3
        for pred in results:
            assert isinstance(pred, np.ndarray)
            np.testing.assert_array_almost_equal(pred, [0.7, 0.3])

    def test_call_converts_input_to_numpy(self, client):
        import torch

        images = [torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)]
        results = client(images)
        assert len(results) == 1

    def test_call_empty_batch(self, client):
        results = client([])
        assert len(results) == 0


# ---------------------------------------------------------------------------
# __init__.py exports tests
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_classes_importable(self):
        from checkmaite.core.serving.rayserve import (
            RayImageClassificationClient,
            RayImageClassificationDeployment,
            RayObjectDetectionClient,
            RayObjectDetectionDeployment,
        )

        assert RayObjectDetectionDeployment is not None
        assert RayImageClassificationDeployment is not None
        assert RayObjectDetectionClient is not None
        assert RayImageClassificationClient is not None

    def test_all_exports(self):
        import checkmaite.core.serving.rayserve as rayserve

        expected = {
            "RayImageClassificationClient",
            "RayImageClassificationDeployment",
            "RayObjectDetectionClient",
            "RayObjectDetectionDeployment",
            "print_serve_status",
        }
        assert set(rayserve.__all__) == expected
