from jatic_ri.image_classification.test_stages.impls.dataeval_shift_test_stage import (
    DatasetShiftTestStage,
)

def test_shift_ic_deck_name():
    test_stage = DatasetShiftTestStage()
    assert test_stage._deck == "image_classification_dataset_evaluation"

    test_stage.load_datasets(None, "Dataset1", None, "Dataset2")  # type: ignore
    assert test_stage.cache_id == "shift_ic_Dataset1_Dataset2.json"
