import pytest

from checkmaite.core._plugins import _clear_registry


@pytest.fixture(autouse=True)
def _reset_plugin_registry() -> None:
    _clear_registry()
    yield
    _clear_registry()
