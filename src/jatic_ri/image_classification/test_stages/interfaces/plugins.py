"""Plugins to be used for image classification implementations of TestStage classes
Implementations may use one or more plugins depending on the application.
"""

import maite.protocols.image_classification as ic


# MODELS --------------------------------------------
class SingleModelPlugin:
    """TestStage Plugin for loading a single, maite-compliant model"""

    model: ic.Model
    model_id: str

    def load_model(self, model: ic.Model, model_id: str) -> None:
        """Injest a pre-loaded maite-compliant model."""
        self.model = model
        self.model_id = model_id


class MultiModelPlugin:
    """TestStage Plugin for loading multiple, maite-compliant models"""

    models: dict[str, ic.Model]  # {model_id: model}

    def load_models(self, models: dict[str, ic.Model]) -> None:
        """Injest a pre-loaded maite-compliant set of models."""
        self.models = models


# DATASETS -------------------------------------------
class SingleDatasetPlugin:
    """TestStage Plugin for loading a single, maite-compliant dataset"""

    dataset: ic.Dataset
    dataset_id: str

    def load_dataset(self, dataset: ic.Dataset, dataset_id: str) -> None:
        """Injest a pre-loaded maite-compliant dataset"""
        self.dataset = dataset
        self.dataset_id = dataset_id


class TwoDatasetPlugin:
    """TestStage Plugin for loading two, maite-compliant datasets"""

    dataset_1: ic.Dataset
    dataset_1_id: str
    dataset_2: ic.Dataset
    dataset_2_id: str

    def load_datasets(
        self,
        dataset_1: ic.Dataset,
        dataset_1_id: str,
        dataset_2: ic.Dataset,
        dataset_2_id: str,
    ) -> None:
        """Injest a pre-loaded maite-compliant set of datasets."""
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.dataset_1_id = dataset_1_id
        self.dataset_2_id = dataset_2_id


# METRICS AND THRESHOLD -------------------------------
class MetricPlugin:
    """TestStage Plugin for loading a metric"""

    metric: ic.Metric
    metric_id: str

    def load_metric(self, metric: ic.Metric, metric_id: str) -> None:
        """Injest a pre-loaded, maite-compliant metric"""
        self.metric = metric
        self.metric_id = metric_id
