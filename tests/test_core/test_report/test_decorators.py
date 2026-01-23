"""Tests for report decorator functions."""

import pytest

from jatic_ri.core._utils import MissingDependencyWarning, deprecated, requires_optional_dependency


def test_deprecated_decorator_basic():
    """Test that deprecated decorator warns when function is called."""

    @deprecated()
    def old_function():
        return "result"

    with pytest.warns(DeprecationWarning, match="old_function.*is deprecated"):
        result = old_function()

    assert result == "result"


def test_deprecated_decorator_with_replacement():
    """Test deprecated decorator with replacement suggestion."""

    @deprecated(replacement="new_function")
    def old_function():
        return "result"

    with pytest.warns(DeprecationWarning, match="Use 'new_function' instead"):
        result = old_function()

    assert result == "result"


def test_deprecated_preserves_function_metadata():
    """Test that deprecated decorator preserves function metadata."""

    @deprecated()
    def documented_function():
        """This is a docstring."""
        return 42

    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "This is a docstring."


def test_requires_optional_dependency_missing():
    """Test requires_optional_dependency raises when module is missing."""

    @requires_optional_dependency("nonexistent_module_xyz")
    def function_needs_dependency():
        return "should not reach here"

    with pytest.raises(ImportError, match="nonexistent_module_xyz"):
        function_needs_dependency()


def test_requires_optional_dependency_missing_with_hint():
    """Test requires_optional_dependency includes install hint in error."""

    @requires_optional_dependency("nonexistent_module", install_hint="pip install nonexistent")
    def function_needs_dependency():
        return "should not reach here"

    with pytest.raises(ImportError, match="pip install nonexistent"):
        function_needs_dependency()


def test_requires_optional_dependency_available():
    """Test requires_optional_dependency allows function when module exists."""

    @requires_optional_dependency("os")  # os is always available
    def function_with_os():
        return "success"

    result = function_with_os()
    assert result == "success"


def test_requires_optional_dependency_preserves_metadata():
    """Test that decorator preserves function metadata."""

    @requires_optional_dependency("os")
    def well_documented_function():
        """Important docs here."""
        return True

    assert well_documented_function.__name__ == "well_documented_function"
    assert well_documented_function.__doc__ == "Important docs here."


def test_missing_dependency_warning_is_user_warning():
    """Test that MissingDependencyWarning is a UserWarning subclass."""
    assert issubclass(MissingDependencyWarning, UserWarning)


def test_combined_decorators():
    """Test that deprecated and requires_optional_dependency can be combined."""

    @deprecated(replacement="new_function")
    @requires_optional_dependency("os")
    def old_function_with_dependency():
        return "works"

    with pytest.warns(DeprecationWarning):
        result = old_function_with_dependency()

    assert result == "works"
