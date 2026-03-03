import hashlib
import json

import maite.protocols.image_classification as ic
from nrtk.interop import MAITEImageClassificationAugmentation

from jatic_ri.core._common.nrtk_robustness_capability import (
    NrtkRobustnessBase,
    NrtkRobustnessConfig,
    NrtkRobustnessOutputs,
)
from jatic_ri.core.cached_tasks import evaluate


class NrtkRobustness(NrtkRobustnessBase[ic.Dataset, ic.Model, ic.Metric]):
    """
    Apply realistic image perturbations, evaluate configured metrics, and report performance deltas.

    Iterates over perturbations that mimic real-world conditions, applies them to the dataset, runs the metric
    on each perturbed variant, and generates a report summarizing changes in model performance.
    """

    def _run(
        self,
        models: list[ic.Model],
        datasets: list[ic.Dataset],
        metrics: list[ic.Metric],
        config: NrtkRobustnessConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> NrtkRobustnessOutputs:
        """Run the capability"""
        model = models[0]
        dataset = datasets[0]
        metric = metrics[0]

        perturbations = []
        for perturber in config.perturber_factory:
            # For each NRTK perturber, we want to create a unique augment_id for caching purposes.
            # Hashing the name of the perturber class and all of its instance attriubutes is a fairly robust way to
            # do this.  Note that a perturber algorithm could change between versions of the library, and this would
            # result in invalid cache hits.

            class_name = perturber.__class__.__name__
            # Get all instance attributes
            attrs = perturber.__dict__ if hasattr(perturber, "__dict__") else {}
            perturber_props = {"class_name": class_name, "attributes": attrs}

            # Create a deterministic hash from the properties
            props_str = json.dumps(perturber_props, sort_keys=True, default=str)
            augment_id = hashlib.sha256(props_str.encode()).hexdigest()

            augmentation = MAITEImageClassificationAugmentation(augment=perturber, augment_id=augment_id)
            perturbed_metrics, _, _ = evaluate(
                model=model,
                dataset=dataset,
                metric=metric,
                batch_size=1,
                augmentation=augmentation,
                return_augmented_data=False,
                return_preds=False,
                use_cache=use_prediction_and_evaluation_cache,
            )
            perturbations.append(perturbed_metrics)

        try:
            return_key = metric.return_key  # type: ignore[attr-defined]
        except AttributeError:
            raise AttributeError(
                "Metric does not have a return_key attribute which is required for NrtkRobustness."
            ) from None

        return NrtkRobustnessOutputs(perturbations=perturbations, return_key=return_key)
