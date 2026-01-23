"""Tests for jatic_ri.ui.__init__ module."""

import sys

import pytest


def test_ui_import_without_panel():
    """Test that importing ui without panel installed raises ImportError."""
    # Mock panel to not be available
    panel_module = sys.modules.get("panel")

    # Temporarily remove panel from sys.modules if it exists
    if "panel" in sys.modules:
        sys.modules["panel"] = None

    # Clear the jatic_ri.ui module if already imported
    if "jatic_ri.ui" in sys.modules:
        del sys.modules["jatic_ri.ui"]

    try:
        # This should raise ImportError
        with pytest.raises(ImportError, match="UI components require the 'ui' extra"):
            import jatic_ri.ui  # noqa: F401
    finally:
        # Restore the original state
        if panel_module is not None:
            sys.modules["panel"] = panel_module
        elif "panel" in sys.modules:
            del sys.modules["panel"]


def test_ui_import_with_panel():
    """Test that importing ui with panel installed works."""
    # This test only runs if panel is actually installed
    pytest.importorskip("panel")

    # Should not raise
    import jatic_ri.ui  # noqa: F401
