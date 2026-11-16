"""The spotcheck extra's dependency contract, checked without the extra installed."""

import importlib.metadata
import sys
import warnings

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from checkmaite.core.component_checks import check_components


def _requirements(extra: str | None) -> dict[str, Requirement]:
    """Requirements in checkmaite's installed metadata: the core ones, or those only ``extra`` adds."""
    requirements = {}
    for line in importlib.metadata.requires("checkmaite") or []:
        requirement = Requirement(line)
        in_core = requirement.marker is None or requirement.marker.evaluate({"extra": ""})
        if extra is None and in_core:
            requirements[requirement.name] = requirement
        elif extra is not None and not in_core and requirement.marker.evaluate({"extra": extra}):
            requirements[requirement.name] = requirement
    return requirements


def test_spotcheck_extra_requires_maite_0_10_and_beartype():
    extra = _requirements("spotcheck")
    assert set(extra) == {"beartype", "maite"}
    assert Version("0.9.5") not in extra["maite"].specifier
    assert Version("0.10.0") in extra["maite"].specifier
    assert Version("0.11.0") not in extra["maite"].specifier


def test_core_install_requires_maite_0_10():
    core = _requirements(None)
    assert Version("0.9.5") not in core["maite"].specifier
    assert Version("0.10.3") in core["maite"].specifier


def test_maite_without_spotcheck_gives_the_install_hint(monkeypatch, fake_od_dataset_default):
    # MAITE 0.9 has no maite.spotcheck module. A None entry in sys.modules makes the import fail the same way.
    monkeypatch.setitem(sys.modules, "maite.spotcheck", None)
    with pytest.raises(ImportError, match=r"install CheckMAITE with the spotcheck extra \(checkmaite\[spotcheck\]\)"):
        check_components(task="object_detection", dataset=fake_od_dataset_default)


def test_maite_without_beartype_gives_the_install_hint(monkeypatch, fake_od_dataset_default):
    # MAITE 0.10 without beartype: maite.spotcheck imports but defines none of its names.
    import maite

    if not hasattr(maite, "__version__") or Version(maite.__version__) < Version("0.10"):
        pytest.skip("needs MAITE 0.10")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import maite.spotcheck  # load it so monkeypatch restores the real module afterwards
    monkeypatch.setitem(sys.modules, "beartype", None)
    monkeypatch.delitem(sys.modules, "maite.spotcheck")
    monkeypatch.delattr(maite, "spotcheck")
    with pytest.raises(ImportError, match=r"install CheckMAITE with the spotcheck extra \(checkmaite\[spotcheck\]\)"):
        check_components(task="object_detection", dataset=fake_od_dataset_default)
