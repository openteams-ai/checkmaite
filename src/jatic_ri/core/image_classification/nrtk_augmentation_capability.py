import maite.protocols.image_classification as ic
from nrtk.interop.maite.interop.image_classification.augmentation import JATICClassificationAugmentation

from jatic_ri.core._common.nrtk_augmentation_capability import (
    NrtkAugmentationBase,
    NrtkAugmentationConfig,
    NrtkAugmentationOutputs,
)
from jatic_ri.core.cached_tasks import evaluate


class NrtkAugmentation(NrtkAugmentationBase[ic.Dataset, ic.Model, ic.Metric]):
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
        config: NrtkAugmentationConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> NrtkAugmentationOutputs:
        """Run the capability"""
        model = models[0]
        dataset = datasets[0]
        metric = metrics[0]

        perturbations = []
        for perturber in config.perturber_factory:
            augmentation = JATICClassificationAugmentation(augment=perturber, augment_id="JATICClassification")
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
                "Metric does not have a return_key attribute which is required for NrtkAugmentation."
            ) from None

        return NrtkAugmentationOutputs(perturbations=perturbations, return_key=return_key)
