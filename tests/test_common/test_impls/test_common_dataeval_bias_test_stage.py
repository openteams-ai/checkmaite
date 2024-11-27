import pytest
from pathlib import Path

import pandas as pd
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import create_text_data_slide
from jatic_ri.image_classification.test_stages.impls.dataeval_bias_test_stage import DatasetBiasTestStage


class TestCommonBiasUtilityFunctions:
    """Test private helper functions used by Bias"""

    @pytest.mark.parametrize("with_table", [True, False])
    def test_create_text_data_slide(self, with_table: bool):
        """Tests TextData arguments are correctly populated with and without a DataFrame"""
        table = pd.DataFrame({"dummy": [0]}) if with_table else None
        result_template = create_text_data_slide(
            deck="DECK",
            title="TITLE",
            heading="HEADING",
            text=["A", "B", "C"],
            table=table,
        )
        assert result_template["deck"] == "DECK"
        assert result_template["layout_name"] == "TextData"

        layout_args = result_template["layout_arguments"]
        assert layout_args["title"] == "TITLE"
        assert layout_args["text_column_heading"] == "HEADING"
        assert layout_args["text_column_half"]
        assert isinstance(layout_args["text_column_body"], list)
        assert ("data_column_table" in layout_args) == with_table

        create_deck(
            [result_template], path=Path("artifacts"), deck_name=f"test_create_text_data_slide_table={str(with_table)}"
        )
