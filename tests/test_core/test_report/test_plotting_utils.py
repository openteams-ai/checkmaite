from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from gradient.templates_and_layouts.create_deck import create_deck

from jatic_ri.core.report._gradient import (
    create_section_by_item_slide,
)
from jatic_ri.core.report._plotting_utils import (
    create_metric_dataframe_data,
    get_cutoff_values,
    get_optimal_subplot_layout,
    prepare_single_histogram,
    save_figure_to_tempfile,
    split_into_chunks,
)


@pytest.fixture(scope="module")
def fake_image() -> str:
    image = np.ones((28, 28, 3), dtype=int) * 200
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    return save_figure_to_tempfile(fig)


class TestCommonBiasUtilityFunctions:
    """Test private helper functions used by Bias"""

    @pytest.mark.parametrize("with_table", [True, False])
    def test_create_section_by_item_slide(self, with_table: bool, fake_image):
        """Tests SectionByItem arguments are correctly populated with and without a DataFrame"""
        if with_table:
            table = pd.DataFrame({"dummy": [0]})
            image_path = None
        else:
            table = None
            image_path = Path(fake_image)

        result_template = create_section_by_item_slide(
            deck="DECK",
            title="TITLE",
            heading="HEADING",
            text=["A", "B", "C"],
            table=table,
            image_path=image_path,
        )
        assert result_template["deck"] == "DECK"
        assert result_template["layout_name"] == "SectionByItem"

        layout_args = result_template["layout_arguments"]
        assert layout_args["title"] == "TITLE"
        assert layout_args["line_section_heading"] == "HEADING"
        assert layout_args["line_section_half"]
        assert isinstance(layout_args["line_section_body"], list)
        if with_table:
            assert isinstance(layout_args["item_section_body"], pd.DataFrame)
        else:
            assert isinstance(layout_args["item_section_body"], Path)

        create_deck(
            [result_template], path=Path("artifacts"), deck_name=f"test_create_text_data_slide_table={str(with_table)}"
        )


def test_single_histogram_constant_array():
    result = prepare_single_histogram("width", np.array([42, 42, 42]), 100, 0)
    assert result[1].sum() == 3
    assert result[3] == 41.5
    assert result[4] == 42.5


def test_single_histogram_few_uniques():
    data = np.arange(10)
    result = prepare_single_histogram("width", data, 8, 1)
    assert len(result[2]) == 11  # bins = 10 → 11 edges


def test_single_histogram_moderate_uniques():
    data = np.arange(11, 25)
    result = prepare_single_histogram("width", data, 20, 15)
    assert len(result[2]) == 15  # bins = 14 → 15 edges


def test_single_histogram_many_uniques():
    data = np.arange(50)
    result = prepare_single_histogram("width", data, 48, 2)
    assert len(result[2]) == 31  # bins >=30 → 31 edges (as max 30 bins)


def test_cutoff_normal_distribution():
    rng = np.random.Generator(np.random.PCG64())
    data = rng.normal(100, 10, 1000)
    top, bot = get_cutoff_values(data)
    assert top > bot
    assert bot >= 0


def test_cutoff_near_zero():
    data = np.array([10, 7, 4])
    top, bot = get_cutoff_values(data)
    assert bot == 0  # forced to 0
    assert np.isclose(top, 14.3484)


def test_cutoff_negative_values():
    data = np.array([-10, -10, -10])
    top, bot = get_cutoff_values(data)
    assert bot == 0  # because std is 0
    assert top == -10


def test_cutoff_large_range():
    data = np.linspace(0, 1000, 10000)
    top, bot = get_cutoff_values(data)
    assert top > 1000 * 0.9  # well into the upper range
    assert bot < 100  # lower but not forced to 0


def test_subplot_layout_single_plot():
    assert get_optimal_subplot_layout(1) == (1, 3)


def test_subplot_layout_perfect_grid():
    assert get_optimal_subplot_layout(12) == (3, 4)


def test_subplot_layout_minimize_empty():
    assert get_optimal_subplot_layout(6) == (2, 3)


def test_subplot_layout_close_to_3_rows():
    assert get_optimal_subplot_layout(15) == (3, 5)


def test_subplot_layout_many_plots():
    rows, cols = get_optimal_subplot_layout(50)
    assert rows * cols >= 50
    assert cols in [3, 4, 5, 6]


def test_split_chunks_perfect_division():
    result = split_into_chunks({"a": 1, "b": 2, "c": 3, "d": 4}, [2])
    assert result == [["a", "b"], ["c", "d"]]


def test_split_chunks_non_divisible():
    result = split_into_chunks({"a": 1, "b": 2, "c": 3}, [2])
    assert result == [["a", "b"], ["c"]]


def test_split_chunks_less_than_chunk_size():
    result = split_into_chunks({"a": 1}, [5])
    assert result == [["a"]]


def test_split_chunks_empty_dict():
    assert split_into_chunks({}, [3]) == []


def test_create_metric_df_images():
    data = {"blur": list(range(10)), "noise": list(range(5))}
    metric_df = create_metric_dataframe_data(True, ["blur"], data, 20)
    assert "Extreme Blur" in metric_df.columns
    assert metric_df.loc[0, "Extreme Blur"] == "50.00%"


def test_create_metric_df_targets():
    data = {"shadow": list(range(8)), "warp": list(range(2))}
    metric_df = create_metric_dataframe_data(False, ["warp"], data, 10)
    assert "Extreme Warp" in metric_df.columns
    assert metric_df.loc[1, "Extreme Warp"] == "2"


def test_create_metric_df_subset_filtering():
    data = {"shadow": list(range(3)), "darkness": list(range(3))}
    metric_df = create_metric_dataframe_data(True, ["shadow"], data, 6)
    assert "Extreme Darkness" not in metric_df.columns
    assert "Extreme Shadow" in metric_df.columns


def test_create_metric_df_empty_subset():
    metric_df = create_metric_dataframe_data(True, [], {}, 1)
    assert metric_df.shape[1] == 1
