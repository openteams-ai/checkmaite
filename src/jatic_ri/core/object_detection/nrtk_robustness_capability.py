import maite.protocols.object_detection as od
from nrtk.interop.maite.interop.object_detection.augmentation import JATICDetectionAugmentation

from jatic_ri.core._common.nrtk_robustness_capability import (
    NrtkRobustnessBase,
    NrtkRobustnessConfig,
    NrtkRobustnessOutputs,
)
from jatic_ri.core.cached_tasks import evaluate


class NrtkRobustness(NrtkRobustnessBase[od.Dataset, od.Model, od.Metric]):
    """
    Augmentation capability that applies realistic image perturbations, evaluates a configured metric, and reports
    performance deltas.

    Iterates over perturbations that mimic real-world conditions, applies them to the dataset, runs the metric
    on each perturbed variant, and generates a report summarizing changes in model performance.
    """

    def _run(
        self,
        models: list[od.Model],
        datasets: list[od.Dataset],
        metrics: list[od.Metric],
        config: NrtkRobustnessConfig,
        use_prediction_and_evaluation_cache: bool,
    ) -> NrtkRobustnessOutputs:
        """Run the capability"""
        model = models[0]
        dataset = datasets[0]
        metric = metrics[0]

        perturbations = []
        for perturber in config.perturber_factory:
            augmentation = JATICDetectionAugmentation(augment=perturber, augment_id="JATICDetection")
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
