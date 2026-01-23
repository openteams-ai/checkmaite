"""
Unit tests for the MarkdownOutput class.

Tests the markdown report generation functionality including:
- Document initialization and basic structure
- Section and heading management
- Text and list formatting
- Table creation and formatting
- Image embedding
- Metrics and metadata
- Table of contents (manual and auto-generated)
- Input validation and error handling
- Builder pattern and method chaining
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from jatic_ri.core.report._markdown import (
    MarkdownOutput,
    _build_display_mapping,
    _copy_local_images_and_build_mapping,
    _ensure_output_dirs,
    _normalize_src_path,
    _rewrite_markdown_paths,
    _unique_destination,
    create_markdown_output,
)


class TestMarkdownOutputInit:
    """Tests for MarkdownOutput initialization."""

    def test_init_with_title(self):
        """Test basic initialization with a title."""
        md = MarkdownOutput("Test Report")
        assert md.title == "Test Report"
        output = md.render()
        assert output.startswith("# Test Report\n")

    def test_str_method(self):
        """Test __str__ returns the same as render()."""
        md = MarkdownOutput("My Report")
        assert str(md) == md.render()

    def test_empty_tracked_sections_on_init(self):
        """Test that tracked sections list is empty on initialization."""
        md = MarkdownOutput("Report")
        assert md._tracked_sections == []


class TestMarkdownOutputSections:
    """Tests for section and heading management."""

    def test_add_section_default_level(self):
        """Test adding a section with default level (2)."""
        md = MarkdownOutput("Report")
        md.add_section("Introduction")
        output = md.render()
        assert "## Introduction" in output

    def test_add_section_custom_level(self):
        """Test adding sections with different levels."""
        md = MarkdownOutput("Report")
        md.add_section("Level 1", level=1)
        md.add_section("Level 2", level=2)
        md.add_section("Level 3", level=3)
        md.add_section("Level 6", level=6)

        output = md.render()
        assert "# Level 1" in output
        assert "## Level 2" in output
        assert "### Level 3" in output
        assert "###### Level 6" in output

    def test_add_section_string_level(self):
        """Test adding sections with string level shortcuts."""
        md = MarkdownOutput("Report")
        md.add_section("Heading 1", level="h1")
        md.add_section("Heading 2", level="h2")
        md.add_section("Heading 3", level="h3")
        md.add_section("Heading 6", level="h6")

        output = md.render()
        assert "# Heading 1" in output
        assert "## Heading 2" in output
        assert "### Heading 3" in output
        assert "###### Heading 6" in output

    def test_add_section_string_level_case_insensitive(self):
        """Test that string level shortcuts are case-insensitive."""
        md = MarkdownOutput("Report")
        md.add_section("Upper", level="H2")
        md.add_section("Lower", level="h3")

        output = md.render()
        assert "## Upper" in output
        assert "### Lower" in output

    def test_add_section_invalid_string_level(self):
        """Test that invalid string level raises ValueError."""
        md = MarkdownOutput("Report")
        with pytest.raises(ValueError, match="Invalid level string 'h7'"):
            md.add_section("Invalid", level="h7")

        with pytest.raises(ValueError, match="Invalid level string 'bad'"):
            md.add_section("Invalid", level="bad")

    def test_add_section_level_clamping(self):
        """Test that level values are clamped to valid range (1-6)."""
        md = MarkdownOutput("Report")
        md.add_section("Too Low", level=0)
        md.add_section("Too High", level=10)

        output = md.render()
        assert "# Too Low" in output  # Clamped to 1
        assert "###### Too High" in output  # Clamped to 6

    def test_add_section_empty_heading(self):
        """Test that empty heading raises ValueError."""
        md = MarkdownOutput("Report")
        with pytest.raises(ValueError, match="Heading cannot be empty"):
            md.add_section("")

        with pytest.raises(ValueError, match="Heading cannot be empty"):
            md.add_section("   ")  # Whitespace only

    def test_add_section_tracks_for_toc(self):
        """Test that sections are tracked for auto-TOC."""
        md = MarkdownOutput("Report")
        md.add_section("First", level=2)
        md.add_section("Second", level=3)

        assert len(md._tracked_sections) == 2
        assert md._tracked_sections[0] == ("First", 2)
        assert md._tracked_sections[1] == ("Second", 3)

    def test_add_subsection(self):
        """Test add_subsection creates level-3 heading."""
        md = MarkdownOutput("Report")
        md.add_subsection("Details")

        output = md.render()
        assert "### Details" in output

    def test_add_section_with_description(self):
        """Test adding section with description."""
        md = MarkdownOutput("Report")
        md.add_section_with_description("Title", "This is the description.")

        output = md.render()
        assert "## Title" in output
        assert "This is the description." in output

    def test_section_chaining(self):
        """Test that add_section returns self for chaining."""
        md = MarkdownOutput("Report")
        result = md.add_section("Section 1")
        assert result is md


class TestMarkdownOutputText:
    """Tests for text content methods."""

    def test_add_text_plain(self):
        """Test adding plain text."""
        md = MarkdownOutput("Report")
        md.add_text("This is a paragraph.")

        output = md.render()
        assert "This is a paragraph." in output

    def test_add_text_bold(self):
        """Test adding bold text."""
        md = MarkdownOutput("Report")
        md.add_text("Important text", bold=True)

        output = md.render()
        assert "**Important text**" in output

    def test_add_text_empty_raises_error(self):
        """Test that empty text raises ValueError."""
        md = MarkdownOutput("Report")
        with pytest.raises(ValueError, match="Text cannot be empty"):
            md.add_text("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            md.add_text("   ")

    def test_add_text_chaining(self):
        """Test that add_text returns self for chaining."""
        md = MarkdownOutput("Report")
        result = md.add_text("Text")
        assert result is md


class TestMarkdownOutputLists:
    """Tests for list formatting methods."""

    def test_add_bulleted_list(self):
        """Test adding a bulleted list."""
        md = MarkdownOutput("Report")
        md.add_bulleted_list(["Item 1", "Item 2", "Item 3"])

        output = md.render()
        assert "- Item 1" in output
        assert "- Item 2" in output
        assert "- Item 3" in output

    def test_add_bulleted_list_nested(self):
        """Test adding nested bulleted lists with indentation."""
        md = MarkdownOutput("Report")
        md.add_bulleted_list(["Top level"])
        md.add_bulleted_list(["Nested level 1"], indent_level=1)
        md.add_bulleted_list(["Nested level 2"], indent_level=2)

        output = md.render()
        assert "- Top level" in output
        assert "  - Nested level 1" in output
        assert "    - Nested level 2" in output

    def test_add_numbered_list(self):
        """Test adding a numbered list."""
        md = MarkdownOutput("Report")
        md.add_numbered_list(["First", "Second", "Third"])

        output = md.render()
        assert "1. First" in output
        assert "2. Second" in output
        assert "3. Third" in output

    def test_add_numbered_list_nested(self):
        """Test adding nested numbered lists with indentation."""
        md = MarkdownOutput("Report")
        md.add_numbered_list(["Step 1"])
        md.add_numbered_list(["Sub-step A", "Sub-step B"], indent_level=1)

        output = md.render()
        assert "1. Step 1" in output
        assert "  1. Sub-step A" in output
        assert "  2. Sub-step B" in output

    def test_list_chaining(self):
        """Test that list methods return self for chaining."""
        md = MarkdownOutput("Report")
        result1 = md.add_bulleted_list(["Item"])
        result2 = md.add_numbered_list(["Item"])
        assert result1 is md
        assert result2 is md


class TestMarkdownOutputTables:
    """Tests for table creation and formatting."""

    def test_add_table_basic(self):
        """Test adding a basic table."""
        md = MarkdownOutput("Report")
        headers = ["Name", "Value"]
        rows = [["Accuracy", "0.95"], ["Precision", "0.92"]]
        md.add_table(headers, rows)

        output = md.render()
        assert "| Name | Value |" in output
        assert "| --- | --- |" in output
        assert "| Accuracy | 0.95 |" in output
        assert "| Precision | 0.92 |" in output

    def test_add_table_with_alignment(self):
        """Test table with column alignment."""
        md = MarkdownOutput("Report")
        headers = ["Left", "Center", "Right"]
        rows = [["L", "C", "R"]]
        align = ["left", "center", "right"]
        md.add_table(headers, rows, align=align)

        output = md.render()
        assert "| :--- | :---: | ---: |" in output

    def test_add_table_escape_pipes(self):
        """Test that pipes in cells are escaped."""
        md = MarkdownOutput("Report")
        headers = ["Column"]
        rows = [["Value | with | pipes"]]
        md.add_table(headers, rows)

        output = md.render()
        assert r"Value \| with \| pipes" in output

    def test_add_table_no_headers_raises_error(self):
        """Test that table with no headers raises ValueError."""
        md = MarkdownOutput("Report")
        with pytest.raises(ValueError, match="Table must have at least one header"):
            md.add_table([], [])

    def test_add_table_mismatched_align_raises_error(self):
        """Test that mismatched align length raises ValueError."""
        md = MarkdownOutput("Report")
        headers = ["A", "B"]
        rows = [["1", "2"]]
        align = ["left"]  # Wrong length

        with pytest.raises(ValueError, match="align length .* must match number of headers"):
            md.add_table(headers, rows, align=align)

    def test_add_table_mismatched_row_length_raises_error(self):
        """Test that rows with wrong column count raise ValueError."""
        md = MarkdownOutput("Report")
        headers = ["A", "B"]
        rows = [["1", "2"], ["1"]]  # Second row has wrong length

        with pytest.raises(ValueError, match="Row 1 has 1 columns, expected 2"):
            md.add_table(headers, rows)

    def test_add_table_with_different_types(self):
        """Test table with different value types."""
        md = MarkdownOutput("Report")
        headers = ["String", "Int", "Float"]
        rows = [["text", 42, 3.14]]
        md.add_table(headers, rows)

        output = md.render()
        assert "| text | 42 | 3.14 |" in output

    def test_table_chaining(self):
        """Test that add_table returns self for chaining."""
        md = MarkdownOutput("Report")
        result = md.add_table(["H"], [["V"]])
        assert result is md


class TestMarkdownOutputImages:
    """Tests for image embedding."""

    def test_add_image_with_path(self):
        """Test adding an image with a path."""
        md = MarkdownOutput("Report")
        md.add_image("path/to/image.png", alt_text="Alt text")

        output = md.render()
        assert "![Alt text](path/to/image.png)" in output

    def test_add_image_with_caption(self):
        """Test adding an image with caption."""
        md = MarkdownOutput("Report")
        md.add_image("image.png", alt_text="Alt", caption="Figure 1: Example")

        output = md.render()
        assert "![Alt](image.png)" in output
        assert "*Figure 1: Example*" in output

    def test_add_image_with_pathlib_path(self):
        """Test adding an image with Path object."""
        md = MarkdownOutput("Report")
        path = Path("images/chart.png")
        md.add_image(path)

        output = md.render()
        assert "![](images/chart.png)" in output

    def test_image_chaining(self):
        """Test that add_image returns self for chaining."""
        md = MarkdownOutput("Report")
        result = md.add_image("test.png")
        assert result is md


class TestMarkdownOutputMetrics:
    """Tests for metrics and metadata."""

    def test_add_metric_basic(self):
        """Test adding a basic metric."""
        md = MarkdownOutput("Report")
        md.add_metric("Accuracy", 0.95)

        output = md.render()
        assert "- **Accuracy**: 0.95" in output

    def test_add_metric_with_format_spec(self):
        """Test metric with format specification."""
        md = MarkdownOutput("Report")
        md.add_metric("Score", 0.123456, format_spec=".2f")

        output = md.render()
        assert "- **Score**: 0.12" in output

    def test_add_metric_invalid_format_spec(self):
        """Test that invalid format spec raises ValueError."""
        md = MarkdownOutput("Report")
        with pytest.raises(ValueError, match="Invalid format_spec"):
            md.add_metric("Value", "text", format_spec=".2f")

    def test_add_metrics_list(self):
        """Test adding multiple metrics as a list."""
        md = MarkdownOutput("Report")
        metrics = {"Accuracy": 0.95, "Precision": 0.92, "Recall": 0.88}
        md.add_metrics_list(metrics)

        output = md.render()
        assert "- **Accuracy**: 0.95" in output
        assert "- **Precision**: 0.92" in output
        assert "- **Recall**: 0.88" in output

    def test_metric_chaining(self):
        """Test that metric methods return self for chaining."""
        md = MarkdownOutput("Report")
        result1 = md.add_metric("M", 1)
        result2 = md.add_metrics_list({"A": 1})
        assert result1 is md
        assert result2 is md


class TestMarkdownOutputDividers:
    """Tests for dividers and separators."""

    def test_add_horizontal_rule(self):
        """Test adding a horizontal rule."""
        md = MarkdownOutput("Report")
        md.add_horizontal_rule()

        output = md.render()
        assert "---" in output

    def test_add_section_divider(self):
        """Test adding a section divider with extra spacing."""
        md = MarkdownOutput("Report")
        md.add_section_divider()

        output = md.render()
        # Section divider has blank lines around it
        # After title there's already a blank line, so pattern is \n\n---\n
        assert "\n\n---\n" in output
        # Verify the divider is present
        assert "---" in output

    def test_divider_chaining(self):
        """Test that divider methods return self for chaining."""
        md = MarkdownOutput("Report")
        result1 = md.add_horizontal_rule()
        result2 = md.add_section_divider()
        assert result1 is md
        assert result2 is md


class TestMarkdownOutputTableOfContents:
    """Tests for table of contents generation."""

    def test_add_toc_manual(self):
        """Test adding a manual table of contents."""
        md = MarkdownOutput("Report")
        sections = ["Introduction", "Methods", "Results"]
        md.add_toc(sections)

        output = md.render()
        assert "## Table of Contents" in output
        assert "1. [Introduction](#introduction)" in output
        assert "2. [Methods](#methods)" in output
        assert "3. [Results](#results)" in output

    def test_add_auto_toc(self):
        """Test automatic TOC generation from tracked sections."""
        md = MarkdownOutput("Report")
        md.add_section("Introduction", level=2)
        md.add_section("Methods", level=2)
        md.add_subsection("Data Collection")
        md.add_section("Results", level=2)

        md.add_auto_toc()

        output = md.render()
        assert "## Table of Contents" in output
        assert "[Introduction](#introduction)" in output
        assert "[Methods](#methods)" in output
        assert "[Data Collection](#data-collection)" in output
        assert "[Results](#results)" in output

    def test_add_auto_toc_with_custom_title(self):
        """Test auto-TOC with custom title."""
        md = MarkdownOutput("Report")
        md.add_section("Section 1")
        md.add_auto_toc(title="Contents")

        output = md.render()
        assert "## Contents" in output

    def test_add_auto_toc_with_max_level(self):
        """Test auto-TOC filtering by max level."""
        md = MarkdownOutput("Report")
        md.add_section("Level 1", level=1)
        md.add_section("Level 2", level=2)
        md.add_section("Level 3", level=3)
        md.add_section("Level 4", level=4)

        md.add_auto_toc(max_level=2)

        output = md.render()
        assert "[Level 1](#level-1)" in output
        assert "[Level 2](#level-2)" in output
        assert "[Level 3](#level-3)" not in output
        assert "[Level 4](#level-4)" not in output

    def test_add_auto_toc_empty_sections(self):
        """Test auto-TOC with no tracked sections."""
        md = MarkdownOutput("Report")
        md.add_auto_toc()

        output = md.render()
        # Should not add TOC if no sections tracked
        assert "## Table of Contents" not in output

    def test_add_auto_toc_indentation(self):
        """Test auto-TOC indents based on heading level."""
        md = MarkdownOutput("Report")
        md.add_section("Main", level=1)
        md.add_section("Sub", level=2)
        md.add_section("SubSub", level=3)

        md.add_auto_toc()

        output = md.render()
        lines = output.split("\n")
        # Find TOC lines
        toc_lines = [line for line in lines if "[Main]" in line or "[Sub]" in line or "[SubSub]" in line]

        assert not toc_lines[0].startswith(" ")  # Level 1, no indent
        assert toc_lines[1].startswith("  ")  # Level 2, 2 spaces
        assert toc_lines[2].startswith("    ")  # Level 3, 4 spaces

    def test_slugify_special_characters(self):
        """Test that _slugify handles special characters."""
        md = MarkdownOutput("Report")

        # Test internal slugify method
        assert md._slugify("Simple Text") == "simple-text"
        assert md._slugify("Text with Numbers 123") == "text-with-numbers-123"
        assert md._slugify("Special!@#$%Characters") == "specialcharacters"
        # Multiple spaces become multiple hyphens (simple replace implementation)
        assert md._slugify("Multiple   Spaces") == "multiple---spaces"

    def test_toc_chaining(self):
        """Test that TOC methods return self for chaining."""
        md = MarkdownOutput("Report")
        result1 = md.add_toc(["Section"])
        md.add_section("Test")
        result2 = md.add_auto_toc()
        assert result1 is md
        assert result2 is md


class TestMarkdownOutputMiscellaneous:
    """Tests for miscellaneous methods."""

    def test_add_blank_line(self):
        """Test adding explicit blank lines."""
        md = MarkdownOutput("Report")
        md.add_text("Line 1")
        md.add_blank_line()
        md.add_blank_line()
        md.add_text("Line 2")

        output = md.render()
        # Multiple blank lines
        assert "Line 1\n\n\n\nLine 2" in output

    def test_add_raw_content(self):
        """Test adding raw markdown content."""
        md = MarkdownOutput("Report")
        md.add_raw("<div>HTML Content</div>")

        output = md.render()
        assert "<div>HTML Content</div>" in output

    def test_misc_chaining(self):
        """Test that misc methods return self for chaining."""
        md = MarkdownOutput("Report")
        result1 = md.add_blank_line()
        result2 = md.add_raw("content")
        assert result1 is md
        assert result2 is md


class TestMarkdownOutputClear:
    """Tests for the clear/reset functionality."""

    def test_clear_resets_content(self):
        """Test that clear() resets the document."""
        md = MarkdownOutput("Report")
        md.add_section("Section 1")
        md.add_text("Some content")

        initial_output = md.render()
        assert "Section 1" in initial_output

        md.clear()
        cleared_output = md.render()

        assert "Section 1" not in cleared_output
        assert cleared_output == "# Report\n"

    def test_clear_preserves_title(self):
        """Test that clear() preserves the title."""
        md = MarkdownOutput("My Report")
        md.add_text("Content")
        md.clear()

        assert md.title == "My Report"
        assert md.render().startswith("# My Report")

    def test_clear_resets_tracked_sections(self):
        """Test that clear() resets tracked sections."""
        md = MarkdownOutput("Report")
        md.add_section("Section 1")
        md.add_section("Section 2")

        assert len(md._tracked_sections) == 2

        md.clear()

        assert len(md._tracked_sections) == 0

    def test_clear_allows_reuse(self):
        """Test that builder can be reused after clear()."""
        md = MarkdownOutput("Report")
        md.add_section("First")
        md.clear()
        md.add_section("Second")

        output = md.render()
        assert "First" not in output
        assert "Second" in output

    def test_clear_chaining(self):
        """Test that clear() returns self for chaining."""
        md = MarkdownOutput("Report")
        result = md.clear()
        assert result is md


class TestMarkdownOutputSaveAndRender:
    """Tests for file I/O operations."""

    def test_render_returns_string(self):
        """Test that render() returns a string."""
        md = MarkdownOutput("Report")
        md.add_text("Content")
        output = md.render()

        assert isinstance(output, str)
        assert len(output) > 0

    def test_save_to_file(self):
        """Test saving markdown to a file."""
        with TemporaryDirectory() as tmpdir:
            md = MarkdownOutput("Test Report")
            md.add_section("Section")
            md.add_text("Content")

            filepath = Path(tmpdir) / "report.md"
            md.save(filepath)

            assert filepath.exists()
            content = filepath.read_text(encoding="utf-8")
            assert "# Test Report" in content
            assert "## Section" in content
            assert "Content" in content

    def test_save_with_string_path(self):
        """Test saving with string path."""
        with TemporaryDirectory() as tmpdir:
            md = MarkdownOutput("Report")
            md.add_text("Test")

            filepath = str(Path(tmpdir) / "report.md")
            md.save(filepath)

            assert Path(filepath).exists()

    def test_save_with_custom_encoding(self):
        """Test saving with custom encoding."""
        with TemporaryDirectory() as tmpdir:
            md = MarkdownOutput("Report")
            md.add_text("Test with émojis 🎉")

            filepath = Path(tmpdir) / "report.md"
            md.save(filepath, encoding="utf-8")

            content = filepath.read_text(encoding="utf-8")
            assert "Test with émojis 🎉" in content


class TestMarkdownOutputChaining:
    """Tests for fluent interface and method chaining."""

    def test_complex_chaining(self):
        """Test complex method chaining."""
        md = MarkdownOutput("Report")
        result = (
            md.add_section("Introduction")
            .add_text("This is the intro.")
            .add_section("Methods")
            .add_bulleted_list(["Step 1", "Step 2"])
            .add_table(["Metric", "Value"], [["Accuracy", "0.95"]])
            .add_horizontal_rule()
        )

        assert result is md
        output = md.render()
        assert "## Introduction" in output
        assert "This is the intro." in output
        assert "## Methods" in output
        assert "- Step 1" in output
        assert "| Metric | Value |" in output
        assert "---" in output

    def test_chaining_preserves_order(self):
        """Test that chained operations preserve order."""
        md = MarkdownOutput("Report")
        md.add_text("First").add_text("Second").add_text("Third")

        output = md.render()
        first_pos = output.index("First")
        second_pos = output.index("Second")
        third_pos = output.index("Third")

        assert first_pos < second_pos < third_pos


class TestMarkdownOutputEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_document(self):
        """Test rendering an empty document (just title)."""
        md = MarkdownOutput("Empty Report")
        output = md.render()

        assert output == "# Empty Report\n"

    def test_unicode_content(self):
        """Test handling of unicode characters."""
        md = MarkdownOutput("Report with émojis 🎉")
        md.add_text("Testing unicode: café, naïve, 日本語")

        output = md.render()
        assert "café" in output
        assert "naïve" in output
        assert "日本語" in output

    def test_multiline_text(self):
        """Test handling of multiline text."""
        md = MarkdownOutput("Report")
        multiline = "Line 1\nLine 2\nLine 3"
        md.add_text(multiline)

        output = md.render()
        assert "Line 1\nLine 2\nLine 3" in output

    def test_empty_list(self):
        """Test behavior with empty lists."""
        md = MarkdownOutput("Report")
        md.add_bulleted_list([])

        output = md.render()
        # Should handle gracefully, just add blank line
        assert "# Report" in output

    def test_single_column_table(self):
        """Test table with single column."""
        md = MarkdownOutput("Report")
        md.add_table(["Column"], [["Value1"], ["Value2"]])

        output = md.render()
        assert "| Column |" in output
        assert "| Value1 |" in output
        assert "| Value2 |" in output

    def test_table_with_empty_cells(self):
        """Test table with empty cell values."""
        md = MarkdownOutput("Report")
        md.add_table(["A", "B"], [["value", ""], ["", "value"]])

        output = md.render()
        assert "| value |  |" in output
        assert "|  | value |" in output


class TestMarkdownImageAssetHelpers:
    """Tests for markdown image asset collection and rewriting helpers.

    These tests validate the small auxiliary functions that:
    - Resolve image sources on disk
    - Copy local images into an output `images/` directory (collision-safe)
    - Rewrite markdown paths for saving and notebook display
    """

    def test_ensure_output_dirs_creates_images_dir(self, tmp_path: Path):
        """Test that _ensure_output_dirs creates <out_dir>/images."""
        out_dir = tmp_path / "out"
        images_dir = _ensure_output_dirs(out_dir)

        assert out_dir.exists()
        assert images_dir.exists()
        assert images_dir == out_dir / "images"

    def test_normalize_src_path_existing_relative(self, tmp_path: Path, monkeypatch):
        """Test resolving a relative path that exists from cwd."""
        monkeypatch.chdir(tmp_path)

        img = tmp_path / "img.png"
        img.write_text("x")

        resolved = _normalize_src_path("img.png", source_root=tmp_path)

        assert resolved is not None
        assert resolved.exists()
        assert resolved.resolve() == img.resolve()

    def test_normalize_src_path_falls_back_to_source_root(self, tmp_path: Path, monkeypatch):
        """Test resolving a path that doesn't exist as-is but exists under source_root."""
        # Simulate cwd != source_root
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(other)

        img_dir = tmp_path / "assets"
        img_dir.mkdir()
        img = img_dir / "img.png"
        img.write_text("x")

        resolved = _normalize_src_path("assets/img.png", source_root=tmp_path)
        assert resolved == img

    def test_normalize_src_path_not_found_returns_none(self, tmp_path: Path):
        """Test that missing paths return None."""
        assert _normalize_src_path("missing.png", source_root=tmp_path) is None

    def test_unique_destination_no_collision(self, tmp_path: Path):
        """Test that _unique_destination returns the original name if not taken."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        src = tmp_path / "img.png"
        src.write_text("x")

        dest = _unique_destination(images_dir, src)
        assert dest == images_dir / "img.png"

    def test_unique_destination_with_collision(self, tmp_path: Path):
        """Test that _unique_destination appends _N suffix if a name is taken."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Existing file consumes the base name
        (images_dir / "img.png").write_text("existing")

        src = tmp_path / "img.png"
        src.write_text("x")

        dest = _unique_destination(images_dir, src)
        assert dest.name.startswith("img_")
        assert dest.suffix == ".png"

    def test_copy_local_images_and_build_mapping_copies_and_maps(self, tmp_path: Path, monkeypatch):
        """Test copying a local image and building the original->images/... rewrite mapping."""
        monkeypatch.chdir(tmp_path)

        img = tmp_path / "img.png"
        img.write_text("data")

        out_dir = tmp_path / "report"
        images_dir = _ensure_output_dirs(out_dir)

        md = "![alt](img.png)\n"
        mapping = _copy_local_images_and_build_mapping(md, images_dir, source_root=tmp_path)

        assert mapping == {"img.png": "images/img.png"}
        assert (images_dir / "img.png").exists()

    def test_copy_local_images_collision_renames_destination(self, tmp_path: Path, monkeypatch):
        """Test that copied images get collision-safe names and mapping points to the renamed file."""
        monkeypatch.chdir(tmp_path)

        img = tmp_path / "img.png"
        img.write_text("new")

        out_dir = tmp_path / "report"
        images_dir = _ensure_output_dirs(out_dir)

        # Pretend a prior image already exists in output
        (images_dir / "img.png").write_text("existing")

        md = "![alt](img.png)\n"
        mapping = _copy_local_images_and_build_mapping(md, images_dir, source_root=tmp_path)

        # Expect a renamed destination like images/img_1.png, images/img_2.png, ...
        assert "img.png" in mapping
        assert mapping["img.png"].startswith("images/img_")
        copied_name = Path(mapping["img.png"]).name
        assert (images_dir / copied_name).exists()

    def test_copy_local_images_skips_remote_and_data_urls(self, tmp_path: Path):
        """Test that remote and data URLs are not copied or rewritten."""
        out_dir = tmp_path / "report"
        images_dir = _ensure_output_dirs(out_dir)

        md = "![r](https://example.com/a.png)\n![d](data:image/png;base64,AAA)\n"

        mapping = _copy_local_images_and_build_mapping(md, images_dir, source_root=tmp_path)
        assert mapping == {}
        assert list(images_dir.iterdir()) == []

    def test_rewrite_markdown_paths_updates_only_mapped_paths(self):
        """Test rewriting markdown paths based on a mapping."""
        md = "![a](a.png)\n![b](b.png)\n"
        mapping = {"a.png": "images/a.png"}

        out = _rewrite_markdown_paths(md, mapping)

        assert "![a](images/a.png)" in out
        assert "![b](b.png)" in out  # unchanged

    def test_build_display_mapping_uses_relpath_from_cwd(self, tmp_path: Path, monkeypatch):
        """Test display mapping converts saved images/... to a cwd-relative path when possible."""
        # Make cwd predictable
        monkeypatch.chdir(tmp_path)

        out_dir = tmp_path / "report"
        img = out_dir / "images" / "a.png"
        img.parent.mkdir(parents=True)
        img.write_text("x")

        saved_md = "![alt](images/a.png)\n"

        mapping = _build_display_mapping(saved_md, out_dir=out_dir)

        # From cwd=tmp_path to tmp_path/report/images/a.png => report/images/a.png
        assert mapping == {"images/a.png": "report/images/a.png"}


def test_create_markdown_output_saves_and_rewrites(tmp_path: Path, monkeypatch):
    """Integration-style test: save markdown, copy image, rewrite paths; optionally display."""
    monkeypatch.chdir(tmp_path)

    from IPython.display import Markdown

    # Patch the display function used by create_markdown_output
    monkeypatch.setattr("jatic_ri.core.report._markdown._ipy_display", lambda *_a, **_kw: None)

    img = tmp_path / "img.png"
    img.write_text("data")

    md = "![alt](img.png)\n"
    out_dir = tmp_path / "report"

    result = create_markdown_output(md, out_dir, display=True)

    saved = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "(images/img.png)" in saved
    assert (out_dir / "images" / "img.png").exists()

    assert isinstance(result, Markdown)
    assert "report/images/img.png" in getattr(result, "data", "")


def test_create_markdown_output_no_display_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    img = tmp_path / "img.png"
    img.write_text("data")

    md = "![alt](img.png)\n"
    out_dir = tmp_path / "report"

    result = create_markdown_output(md, out_dir)  # display defaults to False
    assert result is None

    saved = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "(images/img.png)" in saved
    assert (out_dir / "images" / "img.png").exists()
