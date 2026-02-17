"""
Tests for dataeval_cleaning_capability module.

Tests markdown report generation functions.
"""

import numpy as np
import pytest

from jatic_ri.core._common.dataeval_cleaning_capability import (
    DataevalCleaningDimensionStatsOutputs,
    DataevalCleaningDuplicatesOutputs,
    DataevalCleaningLabelStatsOutputs,
    DataevalCleaningVisualStatsOutputs,
    generate_duplicates_report_md,
    generate_image_outliers_report_md,
    generate_image_property_histograms_report_md,
    generate_image_stats_report_md,
    generate_label_analysis_report_md,
    generate_next_steps_report_md,
    generate_table_of_contents_md,
    generate_target_outliers_report_md,
    generate_target_property_histograms_report_md,
    generate_target_stats_report_md,
)
from jatic_ri.core.report._markdown import MarkdownOutput


@pytest.fixture
def sample_duplicates_output():
    """Create sample duplicates output for testing."""
    return DataevalCleaningDuplicatesOutputs(exact=[[0, 1, 2], [3, 4]], near=[[5, 6], [7, 8, 9]])


@pytest.fixture
def sample_dimension_stats():
    """Create sample dimension stats for testing."""
    from dataeval.types import SourceIndex

    n = 100
    rng = np.random.default_rng(42)
    return DataevalCleaningDimensionStatsOutputs(
        source_index=[SourceIndex(item=i, target=None, channel=None) for i in range(n)],
        object_count=[5] * n,
        image_count=n,
        offset_x=rng.random(n),
        offset_y=rng.random(n),
        width=rng.random(n) * 100 + 200,
        height=rng.random(n) * 100 + 200,
        channels=np.ones(n) * 3,
        size=rng.random(n) * 10000 + 50000,
        aspect_ratio=rng.random(n) * 0.5 + 0.75,
        depth=np.ones(n) * 8,
        center=rng.random(n),
        distance_center=rng.random(n),
        distance_edge=rng.random(n),
        invalid_box=rng.random(n),
    )


@pytest.fixture
def sample_visual_stats():
    """Create sample visual stats for testing."""
    from dataeval.types import SourceIndex

    n = 100
    rng = np.random.default_rng(42)
    return DataevalCleaningVisualStatsOutputs(
        source_index=[SourceIndex(item=i, target=None, channel=None) for i in range(n)],
        object_count=[5] * n,
        image_count=n,
        brightness=rng.random(n) * 255,
        contrast=rng.random(n) * 100,
        darkness=rng.random(n) * 255,
        sharpness=rng.random(n) * 100,
        percentiles=rng.random((n, 3)) * 255,
        missing=rng.random(n),
        zeros=rng.random(n),
    )


@pytest.fixture
def sample_label_stats():
    """Create sample label stats for testing."""
    return DataevalCleaningLabelStatsOutputs(
        label_counts_per_class={0: 100, 1: 150, 2: 75},
        label_counts_per_image=[3, 5, 2, 4],
        image_counts_per_class={0: 50, 1: 75, 2: 40},
        image_indices_per_class={0: [0, 1, 2], 1: [1, 2, 3], 2: [0, 3]},
        image_count=100,
        class_count=3,
        label_count=325,
        class_names=["class0", "class1", "class2"],
    )


def test_generate_table_of_contents_md():
    """Test generate_table_of_contents_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    generate_table_of_contents_md(md)

    output = md.render()
    assert "Table of Contents" in output
    assert len(output) > 0


def test_generate_duplicates_report_md_with_data(sample_duplicates_output):
    """Test generate_duplicates_report_md with duplicate data."""
    md = MarkdownOutput("Test Report")
    generate_duplicates_report_md(md, sample_duplicates_output, 100)

    output = md.render()
    assert "Duplicate" in output
    assert len(output) > 0


def test_generate_duplicates_report_md_no_duplicates():
    """Test generate_duplicates_report_md with no duplicates."""
    empty_output = DataevalCleaningDuplicatesOutputs(exact=[], near=[])
    md = MarkdownOutput("Test Report")
    generate_duplicates_report_md(md, empty_output, 100)

    output = md.render()
    assert len(output) > 0


def test_generate_image_stats_report_md(sample_dimension_stats, sample_visual_stats, sample_label_stats):
    """Test generate_image_stats_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    img_stats = (sample_dimension_stats, sample_visual_stats)
    generate_image_stats_report_md(md, img_stats, sample_label_stats, {0: "class0", 1: "class1", 2: "class2"})

    output = md.render()
    assert "Label" in output or "Image" in output
    assert len(output) > 0


def test_generate_image_outliers_report_md(sample_dimension_stats, sample_visual_stats):
    """Test generate_image_outliers_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    img_outliers = {0: {"width": 0.95}, 1: {"brightness": 0.92}}
    img_stats = (sample_dimension_stats, sample_visual_stats)

    generate_image_outliers_report_md(md, img_outliers, img_stats, 100)

    output = md.render()
    assert "Outlier" in output
    assert len(output) > 0


def test_generate_target_stats_report_md(sample_dimension_stats, sample_visual_stats):
    """Test generate_target_stats_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    box_stats = (sample_dimension_stats, sample_visual_stats)
    generate_target_stats_report_md(md, box_stats, sample_dimension_stats)

    output = md.render()
    assert len(output) > 0


def test_generate_target_outliers_report_md(sample_dimension_stats, sample_visual_stats):
    """Test generate_target_outliers_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    target_outliers = {0: {"aspect_ratio": 0.95}, 1: {"center": 0.93}}
    box_stats = (sample_dimension_stats, sample_visual_stats)

    generate_target_outliers_report_md(md, target_outliers, box_stats, 500)

    output = md.render()
    assert len(output) > 0


def test_generate_next_steps_report_md():
    """Test generate_next_steps_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    generate_next_steps_report_md(md, "jatic_ri")

    output = md.render()
    assert len(output) > 0


def test_generate_image_property_histograms_report_md(sample_dimension_stats, sample_visual_stats):
    """Test generate_image_property_histograms_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    img_stats = (sample_dimension_stats, sample_visual_stats)
    generate_image_property_histograms_report_md(md, img_stats)

    output = md.render()
    assert len(output) > 0


def test_generate_label_analysis_report_md(sample_label_stats):
    """Test generate_label_analysis_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    generate_label_analysis_report_md(md, sample_label_stats, {0: "class0", 1: "class1", 2: "class2"})

    output = md.render()
    assert "Label" in output
    assert len(output) > 0


def test_generate_target_property_histograms_report_md(sample_dimension_stats, sample_visual_stats):
    """Test generate_target_property_histograms_report_md generates proper markdown."""
    md = MarkdownOutput("Test Report")
    box_stats = (sample_dimension_stats, sample_visual_stats)
    generate_target_property_histograms_report_md(md, box_stats, sample_dimension_stats)

    output = md.render()
    assert len(output) > 0


def test_dataeval_cleaning_ic_supports():
    """Test DataevalCleaning support specifications."""
    from jatic_ri.core.capability_core import Number
    from jatic_ri.core.image_classification.dataeval_cleaning_capability import (
        DataevalCleaning,
    )

    capability = DataevalCleaning()

    assert capability.supports_datasets == Number.ONE
    assert capability.supports_models == Number.ZERO
    assert capability.supports_metrics == Number.ZERO
