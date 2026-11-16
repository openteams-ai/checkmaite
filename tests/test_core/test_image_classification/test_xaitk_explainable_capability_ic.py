from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from checkmaite.core.image_classification.xaitk_explainable_capability import (
    XaitkExplainable,
    XaitkExplainableConfig,
)
from checkmaite.core.report._gradient import HAS_GRADIENT
from tests.report_assertions import assert_inline_markdown_report

RISE_ARGS = {
    "name": "XAITKTestStage RISE Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack": {
            "n": 2,
            "s": 4,
            "p1": 0.5,
            "seed": 0,
            "threads": 1,
        },
    },
    "img_batch_size": 1,
}

OCCLUSION_ARGS = {
    "name": "XAITKTestStage PerturbationOcclusion Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based.PerturbationOcclusion",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based.PerturbationOcclusion": {
            "perturber": {
                "type": "xaitk_saliency.impls.perturb_image.sliding_window.SlidingWindow",
                "xaitk_saliency.impls.perturb_image.sliding_window.SlidingWindow": {
                    "window_size": [4, 4],
                    "stride": [4, 4],
                },
            },
            "generator": {
                "type": "xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring.OcclusionScoring",
                "xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring.OcclusionScoring": {},
            },
            "threads": 0,
        },
    },
    "img_batch_size": 1,
}

MC_RISE_ARGS = {
    "name": "XAITKTestStage MC-RISE Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack": {
            "n": 2,
            "s": 4,
            "p1": 0.5,
            "seed": 0,
            "threads": 1,
            "fill_colors": [[255, 0, 0]],
        },
    },
    "img_batch_size": 1,
}


@pytest.fixture
def small_ic_dataset(fake_ic_dataset_default):
    return type(fake_ic_dataset_default)(
        images=fake_ic_dataset_default.images[:2],
        targets=fake_ic_dataset_default.targets[:2],
        datum_metadata=fake_ic_dataset_default.datum_metadata[:2],
        dataset_metadata=fake_ic_dataset_default.metadata,
    )


@pytest.fixture
def fixed_ic_model(fake_ic_model_default):
    """Return an isolated model with the fixture's zero-based class map."""
    return deepcopy(fake_ic_model_default)


@pytest.fixture
def unfixed_ic_model(fake_ic_model_default):
    """Return a model whose index2label keys are 1..N rather than 0..N-1."""
    model = deepcopy(fake_ic_model_default)
    model.metadata["index2label"] = {index + 1: label for index, label in model.metadata["index2label"].items()}
    return model


@pytest.fixture
def rise_config():
    return XaitkExplainableConfig(**RISE_ARGS)


@pytest.fixture
def mc_rise_config():
    return XaitkExplainableConfig(**MC_RISE_ARGS)


@pytest.fixture
def test_run_rise(small_ic_dataset, fixed_ic_model, rise_config) -> Any:
    capability = XaitkExplainable()
    outputs = capability.run(
        use_cache=False,
        models=[fixed_ic_model],
        datasets=[small_ic_dataset],
        config=rise_config,
    )

    assert outputs.model_dump()  # smoke test
    return outputs


@pytest.fixture
def test_run_mc_rise(small_ic_dataset, fixed_ic_model, mc_rise_config) -> Any:
    capability = XaitkExplainable()
    outputs = capability.run(
        use_cache=False,
        models=[fixed_ic_model],
        datasets=[small_ic_dataset],
        config=mc_rise_config,
    )

    assert outputs.model_dump()  # smoke test
    return outputs


@pytest.mark.optional
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables_rise(test_run_rise, small_ic_dataset):
    output = test_run_rise.collect_report_consumables(threshold=0.5)
    assert output  # smoke test
    assert len(output) == len(small_ic_dataset) * small_ic_dataset[0][1].shape[0]


def test_run_and_collect_md_rise(test_run_rise):
    report = test_run_rise.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run_rise.capability_id)


@pytest.mark.optional
@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables_mc_rise(test_run_mc_rise, small_ic_dataset):
    output = test_run_mc_rise.collect_report_consumables(threshold=0.5)
    assert output  # smoke test
    fill_colors = MC_RISE_ARGS["saliency_generator"][MC_RISE_ARGS["saliency_generator"]["type"]]["fill_colors"]
    assert len(output) == len(small_ic_dataset) * small_ic_dataset[0][1].shape[0] * len(fill_colors)


def test_run_and_collect_md_mc_rise(test_run_mc_rise):
    report = test_run_mc_rise.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=test_run_mc_rise.capability_id)


def test_xaitk_capability_rise_unfixed_model(fake_ic_dataset_default, unfixed_ic_model, rise_config) -> None:
    capability = XaitkExplainable()
    run = capability.run(
        use_cache=False,
        models=[unfixed_ic_model],
        datasets=[fake_ic_dataset_default],
        config=rise_config,
    )

    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)


def test_xaitk_capability_mc_rise_unfixed_model(fake_ic_dataset_default, unfixed_ic_model, mc_rise_config) -> None:
    capability = XaitkExplainable()
    run = capability.run(
        use_cache=False,
        models=[unfixed_ic_model],
        datasets=[fake_ic_dataset_default],
        config=mc_rise_config,
    )

    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)


def test_gt_label_uses_class_position_not_key(small_ic_dataset, unfixed_ic_model, rise_config) -> None:
    """The target's argmax is a position on the class axis; map it through the sorted index2label keys."""
    run = XaitkExplainable().run(
        use_cache=False, models=[unfixed_ic_model], datasets=[small_ic_dataset], config=rise_config
    )
    index2label = unfixed_ic_model.metadata["index2label"]
    sorted_labels = [index2label[k] for k in sorted(index2label)]
    assert run.outputs.class_labels == sorted_labels
    for (_, target, _), gt_label, explained in zip(
        small_ic_dataset, run.outputs.gt_labels, run.outputs.explained_class_indices, strict=True
    ):
        position = int(np.argmax(np.asarray(target)))
        assert gt_label == sorted_labels[position]
        assert explained == position


def test_model_output_longer_than_index2label_raises(small_ic_dataset, fixed_ic_model, rise_config) -> None:
    """A model with more outputs than index2label entries gets a clear error, not an IndexError."""
    index2label = fixed_ic_model.metadata["index2label"]
    fixed_ic_model.metadata["index2label"] = {k: index2label[k] for k in sorted(index2label)[:1]}
    with pytest.raises(ValueError, match="Model output has .* classes"):
        XaitkExplainable().run(
            use_cache=False, models=[fixed_ic_model], datasets=[small_ic_dataset], config=rise_config
        )


def test_no_ground_truth_explains_predicted_class(fake_ic_dataset_default, fixed_ic_model, rise_config) -> None:
    """An all-zero target has no ground truth, so the predicted class's map is scored."""
    dataset = type(fake_ic_dataset_default)(
        images=fake_ic_dataset_default.images[:1],
        targets=[np.zeros_like(np.asarray(fake_ic_dataset_default.targets[0]))],
        datum_metadata=fake_ic_dataset_default.datum_metadata[:1],
        dataset_metadata=fake_ic_dataset_default.metadata,
    )
    run = XaitkExplainable().run(use_cache=False, models=[fixed_ic_model], datasets=[dataset], config=rise_config)
    outputs = run.outputs
    assert outputs.gt_labels == [None]
    assert outputs.class_labels[outputs.explained_class_indices[0]] == outputs.pred_labels[0]
    (record,) = run.extract()
    assert record.gt_label is None
    assert record.explained_label == outputs.pred_labels[0]


def test_mc_rise_report_plots_each_class(test_run_mc_rise, monkeypatch) -> None:
    """Each MC-RISE report map is the map of the class it is labelled with, not always class 0."""
    import checkmaite.core.image_classification.xaitk_explainable_capability as module

    shown: list[np.ndarray] = []
    real_imshow = module.plt.imshow

    def spy_imshow(img, *args, **kwargs):
        if kwargs.get("cmap") == "seismic":
            shown.append(np.asarray(img))
        return real_imshow(img, *args, **kwargs)

    monkeypatch.setattr(module.plt, "imshow", spy_imshow)
    test_run_mc_rise.collect_md_report(threshold=0.5)

    expected = [
        np.asarray(sal_maps)[color_idx][class_idx]
        for sal_maps in test_run_mc_rise.outputs.results
        for color_idx in range(np.asarray(sal_maps).shape[0])
        for class_idx in range(np.asarray(sal_maps).shape[1])
    ]
    assert len(shown) == len(expected)
    for got, want in zip(shown, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_mc_rise_records_per_fill_color(test_run_mc_rise) -> None:
    records = test_run_mc_rise.extract()
    n_colors = np.asarray(test_run_mc_rise.outputs.results[0]).shape[0]
    assert len(records) == len(test_run_mc_rise.outputs.results) * n_colors
    assert {r.fill_color_index for r in records} == set(range(n_colors))


def test_record_stats_use_explained_class_map(test_run_rise) -> None:
    outputs = test_run_rise.outputs
    for record in test_run_rise.extract():
        class_idx = outputs.explained_class_indices[record.image_index]
        sal_map = np.asarray(outputs.results[record.image_index])[class_idx]
        assert record.explained_label == outputs.class_labels[class_idx]
        assert record.mean_saliency == pytest.approx(float(np.mean(sal_map)))
        assert record.max_saliency == pytest.approx(float(np.max(sal_map)))


def test_entropy_scores_and_summary(test_run_rise) -> None:
    from xaitk_saliency.impls.saliency_metric.entropy import Entropy

    outputs = test_run_rise.outputs
    for sal_maps, class_idx, entropies, normalized in zip(
        outputs.results,
        outputs.explained_class_indices,
        outputs.saliency_entropy,
        outputs.normalized_saliency_entropy,
        strict=True,
    ):
        sal_map = np.asarray(sal_maps)[class_idx]
        assert entropies == [pytest.approx(Entropy()(sal_map))]
        assert normalized == [pytest.approx(entropies[0] / np.log2(sal_map.size))]
        assert 0.0 <= normalized[0] <= 1.0

    summary = outputs.score_summary["saliency_entropy"]
    flat = [v for row in outputs.saliency_entropy for v in row]
    assert summary.count == len(flat)
    assert summary.mean == pytest.approx(float(np.mean(flat)))
    records = test_run_rise.extract()
    assert [r.saliency_entropy for r in records] == pytest.approx(flat)

    report = test_run_rise.collect_md_report(threshold=0.5)
    assert "Saliency scores" in report.content


def test_run_perturbation_occlusion(small_ic_dataset, fixed_ic_model) -> None:
    """PerturbationOcclusion (sliding window + occlusion scoring) is reachable through config."""
    config = XaitkExplainableConfig(**OCCLUSION_ARGS)
    run = XaitkExplainable().run(use_cache=False, models=[fixed_ic_model], datasets=[small_ic_dataset], config=config)
    n_classes = len(fixed_ic_model.metadata["index2label"])
    for sal_maps, (image, _, _) in zip(run.outputs.results, small_ic_dataset, strict=True):
        assert np.asarray(sal_maps).shape == (n_classes, *np.asarray(image).shape[1:])
    records = run.extract()
    assert len(records) == len(small_ic_dataset)
    assert all(r.saliency_generator_type == "PerturbationOcclusion" for r in records)
    report = run.collect_md_report(threshold=0.5)
    assert_inline_markdown_report(report, capability_id=run.capability_id)


def test_extract_returns_valid_sequential_records(test_run_rise) -> None:
    """extract() returns valid sequential XaitkExplainableRecord records."""
    from checkmaite.core._common.xaitk_explainable_capability import XaitkExplainableRecord

    records = test_run_rise.extract()
    assert len(records) > 0
    assert all(isinstance(r, XaitkExplainableRecord) for r in records)
    assert all(r.gt_label is not None for r in records)
    # OD-specific fields should be None
    assert all(r.detection_index is None for r in records)
    assert all(r.confidence is None for r in records)
    assert all(r.image_id is None for r in records)
    for r in records:
        assert r.mean_saliency == r.mean_saliency  # not NaN
        assert 0.0 <= r.positive_saliency_ratio <= 1.0
        assert r.std_saliency >= 0.0
    assert [r.image_index for r in records] == list(range(len(records)))
