from jatic_ri.object_detection.test_stages import DatasetShiftTestStage


def test_shift_od_deck_name():
    test_stage = DatasetShiftTestStage()
    assert test_stage._deck == "object_detection_dataset_evaluation"
