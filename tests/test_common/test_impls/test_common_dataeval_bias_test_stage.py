import pytest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri._common.test_stages.impls.dataeval_bias_test_stage import create_text_data_slide, create_text_table_data_slide
from jatic_ri.util.utils import save_figure_to_tempfile

@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28,28,3), dtype=int)*200
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return save_figure_to_tempfile(fig)


class TestCommonBiasUtilityFunctions:
    """Test private helper functions used by Bias"""

    @pytest.mark.parametrize("with_table", [True, False])
    def test_create_text_data_slide(self, with_table: bool, fake_image):
        """Tests TextData arguments are correctly populated with and without a DataFrame"""
        if with_table:
            table = pd.DataFrame({"dummy": [0]})
            image_path = None
        else:
            table = None
            image_path = Path(fake_image)

        result_template = create_text_data_slide(
            deck="DECK",
            title="TITLE",
            heading="HEADING",
            text=["A", "B", "C"],
            table=table,
            image_path=image_path,
        )
        assert result_template["deck"] == "DECK"
        assert result_template["layout_name"] == "TextData"

        layout_args = result_template["layout_arguments"]
        assert layout_args["title"] == "TITLE"
        assert layout_args["text_column_heading"] == "HEADING"
        assert layout_args["text_column_half"]
        assert isinstance(layout_args["text_column_body"], list)
        assert ("data_column_table" in layout_args) == with_table
        assert ("data_column_image" in layout_args) != with_table

        create_deck(
            [result_template], path=Path("artifacts"), deck_name=f"test_create_text_data_slide_table={str(with_table)}"
        )
    
    def test_create_text_table_data_slide(self, fake_image):
        """Tests TextTableData arguments are correctly populated"""
        table = pd.DataFrame({"dummy": [0]})
        image_path = Path(fake_image)
        result_template = create_text_table_data_slide(
            deck="DECK",
            title="TITLE",
            heading="HEADING",
            text=["A", "B", "C"],
            table=table,
            image_path=image_path,
        )
        assert result_template["deck"] == "DECK"
        assert result_template["layout_name"] == "TextTableData"

        layout_args = result_template["layout_arguments"] 
        assert layout_args["title"] == "TITLE"
        assert layout_args["text_column_heading"] == "HEADING"
        assert layout_args["text_column_half"]
        assert isinstance(layout_args["text_column_body"], list)
        assert isinstance(layout_args["data_column_table"], pd.DataFrame)
        assert isinstance(layout_args['data_column_image'], Path)

        create_deck([result_template], path=Path("artifacts"), deck_name="test_create_text_table_data_slide")
