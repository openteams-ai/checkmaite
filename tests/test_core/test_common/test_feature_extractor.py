import numpy as np
import torch
from PIL import Image

from jatic_ri.core._common.feature_extractor import (
    load_feature_extractor,
    pca_projector,
    to_unit_interval_01,
)
from jatic_ri.core._types import TorchvisionModelSpec


class TestLoadFeatureExtractor:
    """E2E tests for load_feature_extractor."""

    def test_load_and_extract_embeddings(self):
        """Load a feature extractor and verify it produces embeddings of expected shape."""
        device = torch.device("cpu")
        model_spec = TorchvisionModelSpec(name="resnet18")

        extractor = load_feature_extractor(model_spec, device)

        toy_images = [Image.new("RGB", (64, 64), color=(i * 50, i * 30, i * 20)) for i in range(3)]

        preprocessed = torch.stack([extractor.transforms(img) for img in toy_images])

        with torch.no_grad():
            embeddings = extractor.model(preprocessed)

        assert torch.isfinite(embeddings).all()

    def test_grayscale_image_conversion(self):
        """Verify grayscale images are automatically converted to RGB."""
        device = torch.device("cpu")
        model_spec = TorchvisionModelSpec(name="resnet18")

        extractor = load_feature_extractor(model_spec, device)
        grayscale_img = Image.new("L", (64, 64), color=128)

        # Should not raise - _ensure_rgb converts to RGB
        preprocessed = extractor.transforms(grayscale_img)

        assert preprocessed.shape[0] == 3

    def test_rgba_image_conversion(self):
        """Verify RGBA images are automatically converted to RGB."""
        device = torch.device("cpu")
        model_spec = TorchvisionModelSpec(name="resnet18")

        extractor = load_feature_extractor(model_spec, device)

        rgba_img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 255))

        preprocessed = extractor.transforms(rgba_img)

        assert preprocessed.shape[0] == 3


def test_basic_scaling():
    """Verify basic min-max scaling to [0, 1]."""
    z = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]], dtype=np.float32)

    result = to_unit_interval_01(z)

    assert result.min() >= 0.0
    assert result.max() <= 1.0

    np.testing.assert_allclose(result[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(result[:, 1], [0.0, 0.5, 1.0])


class TestPcaProjector:
    def test_dimension_reduction(self):
        """Verify PCA reduces dimension correctly."""
        rng = np.random.default_rng(42)

        z_ref = rng.standard_normal((100, 50)).astype(np.float32)
        out_dim = 10

        projector = pca_projector(z_ref, out_dim)

        z_new = rng.standard_normal((20, 50)).astype(np.float32)
        z_projected = projector.transform(z_new)

        assert z_projected.shape == (20, out_dim)

    def test_transform_preserves_sample_count(self):
        rng = np.random.default_rng(42)
        z_ref = rng.standard_normal((50, 20)).astype(np.float32)

        out_dim = 5
        projector = pca_projector(z_ref, out_dim=out_dim)

        for n_samples in [1, 10]:
            z_new = rng.standard_normal((n_samples, 20)).astype(np.float32)
            z_projected = projector.transform(z_new)
            assert z_projected.shape == (n_samples, out_dim)

    def test_centering_applied(self):
        """Verify that mean centering is applied during transform."""
        rng = np.random.default_rng(42)

        z_ref = rng.standard_normal((100, 20)).astype(np.float32) + 100.0

        projector = pca_projector(z_ref, out_dim=5)

        np.testing.assert_allclose(projector._pca.mean_, 100.0, atol=0.5)

        # Transform should center the data
        z_centered = projector.transform(z_ref)

        np.testing.assert_allclose(z_centered.mean(), 0.0, atol=0.1)
