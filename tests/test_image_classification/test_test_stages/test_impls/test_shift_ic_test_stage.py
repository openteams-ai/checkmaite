from jatic_ri.image_classification.test_stages.impls.dataeval_shift_test_stage import (
    DatasetShiftTestStage,
)


def test_shift_ic_deck_name():
    test_stage = DatasetShiftTestStage()
    assert test_stage._deck == "image_classification_dataset_evaluation"

    # smoke-test
    test_stage.load_datasets(None, "Dataset1", None, "Dataset2")  # pyright: ignore[reportArgumentType]
