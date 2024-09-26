from typing import Any, Optional  # noqa: D100

import numpy as np
from dataeval._internal.interop import to_numpy
from dataeval.detectors.ood import OOD_AE, OOD_VAEGMM
from dataeval.tensorflow.models import AE, VAEGMM, create_model

from jatic_ri._common.test_stages.interfaces.test_stage import Cache, TestStage
from jatic_ri.object_detection.test_stages.interfaces.plugins import (
    SingleModelPlugin,
    TwoDatasetPlugin,
)
from jatic_ri.util.cache import JSONCache, NumpyEncoder


class DatasetOODTestStage(TestStage[dict[str, Any]], TwoDatasetPlugin, SingleModelPlugin):
    """OODTestStage"""

    outputs: Optional[dict[str, Any]] = None
    cache: Optional[Cache[dict[str, Any]]] = JSONCache(encoder=NumpyEncoder)

    @property
    def cache_id(self) -> str:
        """Unique cache id for output"""
        return f"ood-{self.dataset_1_id}-{self.dataset_2_id}.json"

    def _run(self) -> None:
        """Run OOD detectors"""

        images_1 = np.asarray([to_numpy(data[0]) for data in self.dataset_1])
        images_2 = np.asarray([to_numpy(data[0]) for data in self.dataset_2])

        input_shape = images_1[0].shape

        ood_kwargs = {
            "threshold_perc": 99,
            "epochs": 20,
            "verbose": False,
        }

        detectors = {
            "OOD_AE": OOD_AE(create_model(AE, input_shape)),
            "OOD_VAEGMM": OOD_VAEGMM(create_model(VAEGMM, input_shape)),
        }

        for detector in detectors.values():
            detector.fit(images_1, **ood_kwargs)

        self.outputs = {
            detector_name: detector.predict(images_2).is_ood for detector_name, detector in detectors.items()
        }

    def collect_report_consumables(self) -> list[dict[str, Any]]:
        """Collect OOD results"""

        if not isinstance(self.outputs, dict):
            return []

        return [
            {
                "Method": list(self.outputs),
                "Test statistic": [np.mean(d) for d in self.outputs.values()],
            },
        ]
