"""Tests for optional imports in image_classification.__init__."""

import sys


def test_survivor_import_missing():
    """Test that missing survivor dependency is handled gracefully."""
    # Store original modules
    original_modules = {}
    to_remove = []

    # Remove survivor-related modules
    for mod_name in list(sys.modules.keys()):
        if "survivor" in mod_name.lower():
            original_modules[mod_name] = sys.modules[mod_name]
            to_remove.append(mod_name)

    # Also need to remove checkmaite.core.image_classification to force re-import
    if "checkmaite.core.image_classification" in sys.modules:
        original_modules["checkmaite.core.image_classification"] = sys.modules["checkmaite.core.image_classification"]
        to_remove.append("checkmaite.core.image_classification")

    for mod in to_remove:
        del sys.modules[mod]

    # Mock the import to fail
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if "survivor" in name.lower():
            raise ImportError("Mock: survivor not installed")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mock_import

    try:
        # This should not raise, just log a debug message
        import checkmaite.core.image_classification

        # Survivor should not be available
        assert not hasattr(checkmaite.core.image_classification, "Survivor")
    finally:
        # Restore
        builtins.__import__ = original_import
        for mod_name, mod_obj in original_modules.items():
            sys.modules[mod_name] = mod_obj
