from copy import deepcopy
from typing import Any

import pytest

from checkmaite.core.image_classification.xaitk_explainable_capability import (
    XaitkExplainable,
    XaitkExplainableConfig,
)
from checkmaite.core.report._gradient import HAS_GRADIENT

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
    """
    XAITK requires index2label to be a dict with integer keys from 0..n-1.
    Our test fixture uses string keys starting from '1', so remap here.
    """
    model_copy = deepcopy(fake_ic_model_default)
    model_copy.metadata["index2label"] = {int(i) - 1: label for i, label in model_copy.metadata["index2label"].items()}
    return model_copy


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


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables_rise(test_run_rise, small_ic_dataset):
    output = test_run_rise.collect_report_consumables(threshold=0.5)
    assert output  # smoke test
    assert len(output) == len(small_ic_dataset) * small_ic_dataset[0][1].shape[0]


def test_run_and_collect_md_rise(test_run_rise):
    assert test_run_rise.collect_md_report(threshold=0.5)  # smoke test


@pytest.mark.skipif(not HAS_GRADIENT, reason="gradient package is required for this test")
def test_run_and_collect_consumables_mc_rise(test_run_mc_rise, small_ic_dataset):
    output = test_run_mc_rise.collect_report_consumables(threshold=0.5)
    assert output  # smoke test
    fill_colors = MC_RISE_ARGS["saliency_generator"][MC_RISE_ARGS["saliency_generator"]["type"]]["fill_colors"]
    assert len(output) == len(small_ic_dataset) * small_ic_dataset[0][1].shape[0] * len(fill_colors)


def test_run_and_collect_md_mc_rise(test_run_mc_rise):
    assert test_run_mc_rise.collect_md_report(threshold=0.5)  # smoke test


# --------------------------------------------------------------------
# Keep the original "known-bad" behavior explicitly documented via xfail
# (These use the *unfixed* model fixture on purpose)
# --------------------------------------------------------------------


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_capability_rise_unfixed_model(fake_ic_dataset_default, fake_ic_model_default, rise_config) -> None:
    capability = XaitkExplainable()
    run = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],  # intentionally unfixed
        datasets=[fake_ic_dataset_default],
        config=rise_config,
    )

    # if it ever stops erroring, these should succeed and xfail will alert us
    _ = run.collect_report_consumables(threshold=0.5)
    assert run.collect_md_report(threshold=0.5)


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_capability_mc_rise_unfixed_model(fake_ic_dataset_default, fake_ic_model_default, mc_rise_config) -> None:
    capability = XaitkExplainable()
    run = capability.run(
        use_cache=False,
        models=[fake_ic_model_default],  # intentionally unfixed
        datasets=[fake_ic_dataset_default],
        config=mc_rise_config,
    )

    _ = run.collect_report_consumables(threshold=0.5)
    assert run.collect_md_report(threshold=0.5)


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
