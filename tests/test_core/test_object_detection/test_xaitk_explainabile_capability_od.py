import pytest
from torch import as_tensor, equal

from jatic_ri.core.object_detection.xaitk_explainable_capability import XaitkExplainable, XaitkExplainableConfig

ARGS = {
    "name": "XaitkExplainable Example",
    "saliency_generator": {
        "type": "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack",
        "xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise.DRISEStack": {
            "n": 10,
            "s": 8,
            "p1": 0.5,
            "seed": 0,
            "threads": 4,
        },
    },
    "img_batch_size": 1,
}


@pytest.fixture
def test_config():
    return XaitkExplainableConfig(name=ARGS["name"], saliency_generator=ARGS["saliency_generator"])


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_run_and_collect(fake_od_model_default, fake_od_dataset_default, test_config):
    capability = XaitkExplainable()

    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        config=test_config,
    )

    assert output.model_dump()  # smoke test

    output = output.collect_report_consumables(threshold=0.5)

    assert len(output) == len(fake_od_dataset_default) * len(fake_od_dataset_default[0][1].scores)


def test_xaitk_temp_dataset(fake_od_dataset_default, fake_od_model_default):
    temp_dataset = XaitkExplainable().XaitkExplainableDetectionBaselineDataset(
        fake_od_dataset_default, fake_od_model_default, dets_limit=2
    )

    assert len(temp_dataset) == len(fake_od_dataset_default)

    for i in range(len(temp_dataset)):
        assert equal(temp_dataset[i][0], fake_od_dataset_default[i][0])

        dets_i = fake_od_model_default(fake_od_dataset_default[i])[0]
        max_score_i = dets_i.scores.argmax()

        assert equal(as_tensor(temp_dataset[i][1].boxes)[0], dets_i.boxes[max_score_i])
        assert equal(as_tensor(temp_dataset[i][1].labels)[0], dets_i.labels[max_score_i])
        assert equal(as_tensor(temp_dataset[i][1].scores)[0], dets_i.scores[max_score_i])

        assert len(as_tensor(temp_dataset[i][1].boxes)) <= 2
