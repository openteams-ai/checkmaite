import pytest

pytest.importorskip("heart_library")

from checkmaite.core.object_detection.heart_adversarial_capability import HeartAdversarial  # noqa: E402


@pytest.mark.heart
@pytest.mark.unsupported
def test_run_and_collect(fake_od_model_default, fake_od_dataset_default, fake_od_metric_default):
    capability = HeartAdversarial()
    output = capability.run(
        use_cache=False,
        models=[fake_od_model_default],
        datasets=[fake_od_dataset_default],
        metrics=[fake_od_metric_default],
    )

    assert output.model_dump()  # smoke test

    assert output.collect_report_consumables(threshold=0.5)  # smoke test

    assert output.collect_md_report(threshold=0.5)  # smoke test
