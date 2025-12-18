"""
Markdown report generation utilities.

This module provides a builder class for generating structured Markdown reports
in a consistent and maintainable way. Used within the collect_md_report methods.
"""

import os
import re
from collections.abc import Iterable
from pathlib import Path
from shutil import copy2
from typing import Any
from urllib.parse import urlparse

from IPython.display import Markdown, display


# Markdown image syntax: ![alt](path "optional title")
class MarkdownOutput:
    """Builder class for generating structured Markdown reports.

    This class provides a fluent interface for constructing Markdown documents
    with consistent formatting. It handles common markdown elements like headings,
    tables, lists, images, and more.

    Parameters
    ----------
    title : str
        The main title (H1) of the document.

    Examples
    --------
    >>> md = MarkdownOutput("My Report")
    >>> md.add_section(heading="Introduction") \\
    ...   .add_text("This is the introduction.") \\
    ...   .add_section("Results") \\
    ...   .add_table(["Name", "Value"], [["Accuracy", "0.95"]]) \\
    ...   .render()
    """

    def __init__(self, title: str) -> None:
        """Initialize the Markdown builder with a title.

        Parameters
        ----------
        title : str
            The main heading (H1) for the document.
        """
        self.title: str = title
        self._sections: list[str] = [f"# {title}", ""]
        self._tracked_sections: list[tuple[str, int]] = []

    def __str__(self) -> str:
        """Return the rendered markdown string."""
        return self.render()

    @staticmethod
    def _get_alignment(align: str) -> str:
        """Convert alignment specification to markdown separator."""
        if align == "left":
            return ":---"
        if align == "right":
            return "---:"
        if align == "center":
            return ":---:"
        return "---"

    @staticmethod
    def _format_table_cell(value: Any) -> str:
        """Format a table cell and escape markdown-sensitive characters."""
        s = str(value)
        # Escape pipes to avoid breaking the table
        return s.replace("|", r"\|")

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert a heading to a markdown anchor-friendly slug."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9 -]", "", text)
        return text.replace(" ", "-")

    def _append(self, line: str = "") -> None:
        """Append a line to the internal sections list.

        Note: This method returns None and cannot be chained.
        Use public methods for fluent chaining.
        """
        self._sections.append(line)

    def _append_blank(self) -> None:
        """Append a blank line to the internal sections list.

        Note: This method returns None and cannot be chained.
        Use public methods for fluent chaining.
        """
        self._sections.append("")

    def add_section(self, heading: str, level: int | str = 2) -> "MarkdownOutput":
        """Add a section heading.

        Parameters
        ----------
        heading : str
            The text of the heading. Cannot be empty.
        level : int | str, optional
            The heading level (1-6) or string shortcuts ("h1"-"h6"),
            by default 2 (creates ##).

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If heading is empty or level is invalid.
        """
        if not heading or not heading.strip():
            raise ValueError("Heading cannot be empty.")

        # Convert string level to int
        if isinstance(level, str):
            level_map = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
            level_lower = level.lower()
            if level_lower not in level_map:
                raise ValueError(f"Invalid level string '{level}'. Use 'h1'-'h6' or integer 1-6.")
            level = level_map[level_lower]

        level = max(1, min(6, level))  # clamp to valid markdown levels
        self._append(f"{'#' * level} {heading}")
        self._append_blank()

        # Track section for auto-TOC
        self._tracked_sections.append((heading, level))

        return self

    def add_subsection(self, heading: str) -> "MarkdownOutput":
        """Add a level-3 subsection (###).

        Parameters
        ----------
        heading : str
            The text of the subsection heading.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_subsection("Details")
        """
        return self.add_section(heading, level=3)

    def add_text(self, text: str, bold: bool = False) -> "MarkdownOutput":
        """Add a paragraph of text.

        Parameters
        ----------
        text : str
            The text content. Cannot be empty.
        bold : bool, optional
            Whether to make the text bold, by default False.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        content = f"**{text}**" if bold else text
        self._append(content)
        self._append_blank()
        return self

    def add_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        align: list[str] | None = None,
    ) -> "MarkdownOutput":
        """Add a markdown table.

        Parameters
        ----------
        headers : list[str]
            Column headers for the table.
        rows : list[list[Any]]
            Table rows, where each row is a list of cell values.
        align : list[str] | None, optional
            Column alignment specifications ("left", "right", "center"),
            by default None (left-aligned).
        """
        num_cols = len(headers)

        if num_cols == 0:
            raise ValueError("Table must have at least one header column.")

        if align is not None and len(align) != num_cols:
            raise ValueError(f"align length ({len(align)}) must match number of headers ({num_cols}).")

        # Pre-format all cells once for validation and rendering
        formatted_headers = [self._format_table_cell(h) for h in headers]
        formatted_rows: list[list[str]] = []

        for idx, row in enumerate(rows):
            if len(row) != num_cols:
                raise ValueError(f"Row {idx} has {len(row)} columns, expected {num_cols}.")
            formatted_rows.append([self._format_table_cell(cell) for cell in row])

        header_line = "| " + " | ".join(formatted_headers) + " |"
        self._append(header_line)

        # Build separator with alignment
        seps = [self._get_alignment(a) for a in align] if align is not None else ["---"] * num_cols
        sep_line = "| " + " | ".join(seps) + " |"
        self._append(sep_line)

        for formatted_row in formatted_rows:
            row_line = "| " + " | ".join(formatted_row) + " |"
            self._append(row_line)

        self._append_blank()
        return self

    def add_image(
        self,
        path: str | Path,
        alt_text: str = "",
        caption: str | None = None,
    ) -> "MarkdownOutput":
        """Add an image with optional caption.

        Parameters
        ----------
        path : str | Path
            Filesystem path to the image.
        alt_text : str, optional
            Alt text for the image, by default "".
        caption : str | None, optional
            Optional caption rendered as italic text below the image.
        """

        self._append(f"![{alt_text}]({Path(path).as_posix()})")

        if caption:
            self._append(f"*{caption}*")
        self._append_blank()

        return self

    def add_metric(
        self,
        name: str,
        value: str | float,
        format_spec: str = "",
    ) -> "MarkdownOutput":
        """Add a single key-value metric as a bullet point.

        Parameters
        ----------
        name : str
            The metric name.
        value : str | float
            The metric value.
        format_spec : str, optional
            Python format specification for the value, by default "".

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Raises
        ------
        ValueError
            If the format_spec is invalid for the given value.
        """
        if format_spec:
            try:
                formatted_value = f"{value:{format_spec}}"
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid format_spec '{format_spec}' for value {value!r}: {e}") from e
        else:
            formatted_value = str(value)

        self._append(f"- **{name}**: {formatted_value}")
        return self

    def add_metrics_list(self, metrics: dict[str, Any]) -> "MarkdownOutput":
        """Add multiple metrics as bullet points."""
        for name, value in metrics.items():
            self.add_metric(name, value)
        self._append_blank()
        return self

    def add_bulleted_list(self, items: list[str], indent_level: int = 0) -> "MarkdownOutput":
        """Add a bulleted (unordered) list.

        Parameters
        ----------
        items : list[str]
            List items to add.
        indent_level : int, optional
            Indentation level for nested lists (0 = no indent), by default 0.
            Each level adds 2 spaces of indentation.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_bulleted_list(["Item 1", "Item 2"])
        >>> md.add_bulleted_list(["Nested A", "Nested B"], indent_level=1)
        """
        indent = "  " * indent_level
        for item in items:
            self._append(f"{indent}- {item}")
        self._append_blank()
        return self

    def add_numbered_list(self, items: list[str], indent_level: int = 0) -> "MarkdownOutput":
        """Add a numbered (ordered) list.

        Parameters
        ----------
        items : list[str]
            List items to add.
        indent_level : int, optional
            Indentation level for nested lists (0 = no indent), by default 0.
            Each level adds 2 spaces of indentation.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_numbered_list(["Step 1", "Step 2"])
        >>> md.add_numbered_list(["Sub-step A", "Sub-step B"], indent_level=1)
        """
        indent = "  " * indent_level
        for i, item in enumerate(items, 1):
            self._append(f"{indent}{i}. {item}")
        self._append_blank()
        return self

    def add_horizontal_rule(self) -> "MarkdownOutput":
        """Add a horizontal rule (---) within content.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_text("First section")
        >>> md.add_horizontal_rule()
        >>> md.add_text("Second section")
        """
        self._append("---")
        self._append_blank()
        return self

    def add_toc(self, sections: list[str]) -> "MarkdownOutput":
        """Add a table of contents with links to sections.

        Parameters
        ----------
        sections : list[str]
            List of section names to include in the TOC.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        See Also
        --------
        add_auto_toc : Automatically generate TOC from tracked sections.
        """
        self._append("## Table of Contents")
        self._append_blank()

        for i, section in enumerate(sections, 1):
            anchor = self._slugify(section)
            self._append(f"{i}. [{section}](#{anchor})")

        self._append_blank()
        return self

    def add_auto_toc(self, title: str = "Table of Contents", max_level: int = 3) -> "MarkdownOutput":
        """Add an automatically generated table of contents from tracked sections.

        This method generates a TOC based on all sections that have been added
        to the document using add_section() or add_subsection().

        Parameters
        ----------
        title : str, optional
            The title for the table of contents, by default "Table of Contents".
        max_level : int, optional
            Maximum heading level to include (1-6), by default 3.
            For example, max_level=3 includes h1, h2, and h3 headings.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("My Report")
        >>> md.add_section("Introduction")
        >>> md.add_section("Methods")
        >>> md.add_subsection("Data Collection")
        >>> md.add_auto_toc()  # Will include all tracked sections

        Notes
        -----
        Call this method after adding all sections you want included in the TOC.
        """
        if not self._tracked_sections:
            return self

        self._append(f"## {title}")
        self._append_blank()

        counter = 1
        for heading, level in self._tracked_sections:
            if level <= max_level:
                anchor = self._slugify(heading)
                indent = "  " * (level - 1)  # Indent based on level
                self._append(f"{indent}{counter}. [{heading}](#{anchor})")
                counter += 1

        self._append_blank()
        return self

    def add_blank_line(self) -> "MarkdownOutput":
        """Add an explicit blank line for spacing control."""
        self._append_blank()
        return self

    def add_section_with_description(
        self,
        heading: str,
        description: str,
        level: int = 2,
    ) -> "MarkdownOutput":
        """Add a section heading followed by a description paragraph."""
        self.add_section(heading, level=level)
        self.add_text(description)
        self.add_blank_line()
        return self

    def add_section_divider(self) -> "MarkdownOutput":
        """Add a horizontal rule divider between major sections.

        This adds extra blank lines around the rule for visual separation.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_section("Part 1")
        >>> md.add_text("Content...")
        >>> md.add_section_divider()
        >>> md.add_section("Part 2")
        """
        self._append_blank()
        self._append("---")
        self._append_blank()
        return self

    def add_raw(self, content: str) -> "MarkdownOutput":
        """Add raw markdown content directly."""
        self._append(content)
        return self

    def clear(self) -> "MarkdownOutput":
        """Clear the document and reset to initial state with the same title.

        This allows reusing the builder instance for a new document.

        Returns
        -------
        MarkdownOutput
            The builder instance for method chaining.

        Examples
        --------
        >>> md = MarkdownOutput("Report")
        >>> md.add_text("Some content")
        >>> md.clear()  # Reset and start over
        >>> md.add_text("New content")
        """
        self._sections = [f"# {self.title}", ""]
        self._tracked_sections = []
        return self

    def render(self) -> str:
        """Generate the final markdown string.

        Returns
        -------
        str
            The complete markdown document as a string.
        """
        return "\n".join(self._sections)

    def display(self) -> None:
        """Display the rendered markdown in a Jupyter notebook."""
        from IPython.display import Markdown, display

        display(Markdown(self.render()))

    def save(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Write the rendered markdown to a file.

        Parameters
        ----------
        path : str | Path
            Output file path.
        encoding : str, optional
            File encoding, by default "utf-8".
        """
        Path(path).write_text(self.render(), encoding=encoding)


_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+[\"'][^)]+[\"'])?\)")


def _is_remote_url(s: str) -> bool:
    """Return True if `s` looks like an http(s) URL."""
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except (ValueError, AttributeError, TypeError):
        return False


def _is_data_url(s: str) -> bool:
    """Return True for inline base64/data URLs."""
    return s.startswith("data:")


def _ensure_output_dirs(out_dir: Path) -> Path:
    """Create `out_dir` and `out_dir/images` if needed. Return images_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _iter_markdown_images(md: str) -> Iterable[re.Match]:
    """Yield regex matches for markdown image references."""
    return _IMG_RE.finditer(md)


def _normalize_src_path(orig_path: str, *, source_root: Path | None = None) -> Path | None:
    """
    Resolve an image reference into an existing filesystem Path.

    Resolution rules
    ----------------
    1) If the path exists as-is (absolute or relative), use it.
    2) Otherwise, try resolving it relative to `source_root` (defaults to cwd).
    3) If neither exists, return None (skip).
    """
    source_root = source_root or Path.cwd()

    src = Path(orig_path)
    if src.exists():
        return src

    alt = source_root / src
    return alt if alt.exists() else None


def _unique_destination(images_dir: Path, src: Path) -> Path:
    """Pick a destination under `images_dir` that avoids name collisions."""
    dest = images_dir / src.name
    i = 1
    while dest.exists():
        dest = images_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    return dest


def _copy_local_images_and_build_mapping(
    md: str,
    images_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, str]:
    """
    Copy each *local* markdown image into `images_dir` and build a rewrite mapping.

    Returns
    -------
    dict[str, str]
        Maps original markdown path string -> 'images/<dest_name>' for saved markdown.
    """
    orig_to_saved_rel: dict[str, str] = {}

    for m in _iter_markdown_images(md):
        orig = m.group("path")

        # Skip remote/data URLs
        if _is_remote_url(orig) or _is_data_url(orig):
            continue

        src = _normalize_src_path(orig, source_root=source_root)
        if src is None:
            continue

        dest = _unique_destination(images_dir, src)
        copy2(src, dest)

        saved_rel = (Path("images") / dest.name).as_posix()
        orig_to_saved_rel[orig] = saved_rel

    return orig_to_saved_rel


def _rewrite_markdown_paths(md: str, mapping: dict[str, str]) -> str:
    """Rewrite markdown image paths using the captured (path) group."""

    def _repl(m: re.Match) -> str:
        alt = m.group("alt")
        path = m.group("path")
        new_path = mapping.get(path, path)
        return f"![{alt}]({new_path})"

    return _IMG_RE.sub(_repl, md)


def _save_markdown(md: str, out_dir: Path, md_filename: str) -> Path:
    """Write markdown content to `<out_dir>/<md_filename>` and return the path."""
    md_file = out_dir / md_filename
    md_file.write_text(md, encoding="utf-8")
    return md_file


def _build_display_mapping(saved_md: str, *, out_dir: Path) -> dict[str, str]:
    """
    Rewrite 'images/<name>' references for notebook display using cwd-relative paths.

    For each 'images/<name>' link in the saved markdown, compute the on-disk absolute
    path under `out_dir`, then convert it to a relative path from the notebook's cwd.
    If that can't be computed, keep the saved path unchanged.
    """
    mapping: dict[str, str] = {}

    for m in _iter_markdown_images(saved_md):
        saved_path = m.group("path")
        if not saved_path.startswith("images/"):
            continue

        dest_abs = (out_dir / saved_path).resolve()

        try:
            rel_from_cwd = os.path.relpath(dest_abs, start=Path.cwd())
            display_path = Path(rel_from_cwd).as_posix()
        except (ValueError, OSError):
            display_path = saved_path

        # If relpath unexpectedly yields an absolute path, keep saved path.
        if Path(display_path).is_absolute():
            display_path = saved_path

        mapping[saved_path] = display_path

    return mapping


def create_markdown_output(md_report: str, path: str | Path, md_filename: str = "report.md") -> Markdown:
    """
    Save a markdown report and its local image assets into `path`, then display it.

    Steps
    -----
    1) Create `<path>/` and `<path>/images/`.
    2) Copy local images referenced by markdown into `<path>/images/` (collision-safe).
    3) Rewrite markdown so saved links point to `images/<name>`, and save `<md_filename>`.
    4) Rewrite `images/<name>` links again for notebook display so they resolve from cwd.
    5) Display and return the IPython Markdown object.
    """
    out_dir = Path(path)
    images_dir = _ensure_output_dirs(out_dir)

    orig_to_saved_rel = _copy_local_images_and_build_mapping(
        md_report,
        images_dir,
        source_root=Path.cwd(),
    )

    saved_md = _rewrite_markdown_paths(md_report, orig_to_saved_rel)
    _save_markdown(saved_md, out_dir, md_filename)

    saved_rel_to_display = _build_display_mapping(saved_md, out_dir=out_dir)
    displayed_md = _rewrite_markdown_paths(saved_md, saved_rel_to_display)

    md_obj = Markdown(displayed_md)
    display(md_obj)
    return md_obj
