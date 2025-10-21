from jatic_ri.object_detection.test_stages import DatasetShiftTestStage


def test_shift_od_deck_name():
    test_stage = DatasetShiftTestStage()
    assert test_stage._deck == "object_detection_dataset_evaluation"

    # smoke-test
    test_stage.load_datasets(None, "Dataset1", None, "Dataset2")  # pyright: ignore[reportArgumentType]
