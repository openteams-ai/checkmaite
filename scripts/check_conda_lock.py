"""Check if conda lockfile is consistent with pyproject.toml.
Conda lock does not include unsupported and ui optional dependencies
"""

import logging
from pathlib import Path

from conda_lock.conda_lock import DEFAULT_MAPPING_URL
from conda_lock.src_parser import make_lock_spec
from ruamel.yaml import YAML

HERE = __file__ = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[0]
CONDA_LOCKFILE_PATH = REPO_ROOT / "conda-lock.yml"
PYPROJECT_TOML_PATH = REPO_ROOT / "pyproject.toml"
PYPROJECT_CATEGORIES = {"main", "dev"}

logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")


def _get_lockfile_root_packages(parsed_lock_data: dict) -> set[str]:
    """
    Identifies packages that are not dependencies of any other package in parsed_lock_data.

    Args:
        parsed_lock_data: A dictionary representing the parsed YAML content, where 'package' is a list of package
        dictionaries.

    Returns:
        A set of package names that are root nodes.
    """
    if not parsed_lock_data or "package" not in parsed_lock_data:
        return set()

    all_package_names = set()
    all_dependency_names = set()

    packages = parsed_lock_data.get("package", [])

    for pkg in packages:
        if "name" in pkg:
            all_package_names.add(pkg["name"])
        if "dependencies" in pkg and pkg["dependencies"] is not None:
            # Dependencies can be a dictionary or a list of strings
            dependencies = pkg["dependencies"]
            if not isinstance(dependencies, dict):
                raise ValueError(f"Expected dependencies to be a dictionary, got {type(dependencies)}")
            for dep_name in dependencies:
                all_dependency_names.add(dep_name)

    return all_package_names - all_dependency_names


def main() -> bool:
    """
    Check if conda-lock.yml is in sync with pyproject.toml dependencies.

    Returns:
        True if validation passes, False if there are issues.
    """
    try:
        with open(CONDA_LOCKFILE_PATH) as f:
            lock_data = yaml.load(f)
    except FileNotFoundError:
        logger.exception(f"Error: {CONDA_LOCKFILE_PATH} not found")
        return False
    except OSError as e:
        logger.exception(f"Error reading {CONDA_LOCKFILE_PATH}: {e}")
        return False

    # create the lock spec
    try:
        lock_spec = make_lock_spec(
            src_files=[
                PYPROJECT_TOML_PATH,
            ],
            mapping_url=DEFAULT_MAPPING_URL,
            filtered_categories=PYPROJECT_CATEGORIES,
        )
    except (FileNotFoundError, OSError) as e:
        logger.exception(f"Error creating lock spec from {PYPROJECT_TOML_PATH}: {e}")
        return False

    all_lockfile_packages = {pkg["name"] for pkg in lock_data.get("package", []) if "name" in pkg}

    lockfile_root_packages = _get_lockfile_root_packages(lock_data)
    logger.debug(f"Root packages: {lockfile_root_packages}")

    lock_spec_packages = {d.name for d in lock_spec.dependencies["linux-64"]}

    # ensure every dependency in the spec is in the lockfile
    if not lock_spec_packages.issubset(all_lockfile_packages):
        missing_packages = lock_spec_packages - all_lockfile_packages
        logger.exception(
            f"{CONDA_LOCKFILE_PATH.name} is missing packages required "
            f"by {PYPROJECT_TOML_PATH.name}: {missing_packages}. "
            "Run `make conda-lock` to update the lockfile."
        )
        return False

    # ensure no extra packages in the lockfile
    if not lockfile_root_packages.issubset(lock_spec_packages):
        extra_packages = lockfile_root_packages - lock_spec_packages
        logger.exception(
            f"{CONDA_LOCKFILE_PATH.name} contains packages not required by the lockspec: {extra_packages} "
            "Run `make conda-lock` to update the lockfile."
        )
        return False

    logger.info(f"{CONDA_LOCKFILE_PATH.name} successfully validated")
    return True


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check if conda lockfile is consistent with pyproject.toml")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    success = main()
    sys.exit(0 if success else 1)
