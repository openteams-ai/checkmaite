from jatic_ri.image_classification.test_stages import DatasetShiftTestStage


def test_shift_ic_deck_name():
    test_stage = DatasetShiftTestStage()
    assert test_stage._deck == "image_classification_dataset_evaluation"
