from copy import deepcopy

import pytest

from jatic_ri.core.image_classification.xaitk_explainable_capability import XaitkExplainable, XaitkExplainableConfig

RISE_ARGS = {
    "name": "XAITKTestStage RISE Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
        },
    },
    "img_batch_size": 1,
}

MC_RISE_ARGS = {
    "name": "XAITKTestStage MC-RISE Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack",
        "xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise.MCRISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
            "fill_colors": [[255, 0, 0], [0, 255, 0]],
        },
    },
    "img_batch_size": 1,
}


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_capability_rise(fake_ic_dataset_default, fake_ic_model_default) -> None:
    capability = XaitkExplainable()

    config = XaitkExplainableConfig(**RISE_ARGS)

    run = capability.run(
        use_cache=False, models=[fake_ic_model_default], datasets=[fake_ic_dataset_default], config=config
    )
    output = run.collect_report_consumables(threshold=0.5)

    assert len(output) == len(fake_ic_dataset_default) * fake_ic_dataset_default[0][1].shape[0]

    md = run.collect_md_report(threshold=0.5)
    assert md  # smoke test


@pytest.mark.xfail(reason="XAITK errors when model 'index2label' keys are not integers from 0 to n-1 consecutively")
def test_xaitk_capability_mc_rise(fake_ic_dataset_default, fake_ic_model_default) -> None:
    capability = XaitkExplainable()

    config = XaitkExplainableConfig(**MC_RISE_ARGS)

    run = capability.run(
        use_cache=False, models=[fake_ic_model_default], datasets=[fake_ic_dataset_default], config=config
    )
    output = run.collect_report_consumables(threshold=0.5)

    assert len(output) == len(fake_ic_dataset_default) * fake_ic_dataset_default[0][1].shape[0] * 2

    md = run.collect_md_report(threshold=0.5)
    assert md  # smoke test


def test_run_and_collect(fake_ic_dataset_default, fake_ic_model_default):
    capability = XaitkExplainable()

    model_copy = deepcopy(fake_ic_model_default)
    # xaitk requires index2label to be a dict with integer keys from 0 to n-1
    # test fixture uses string keys starting from '1' so we remap them here
    model_copy.metadata["index2label"] = {int(i) - 1: label for i, label in model_copy.metadata["index2label"].items()}

    output = capability.run(use_cache=False, models=[model_copy], datasets=[fake_ic_dataset_default])

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
