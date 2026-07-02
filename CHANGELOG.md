# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-07-02

### Changed
- Switched production PyPI publishing to API-token authentication while self-managed GitLab Trusted Publishing onboarding is pending.

## [0.2.1] - 2026-07-02

### Added
- JATIC ONNX model wrappers
- Runtime shape validation for the image-classification metric wrapper
- Provenance tracking in analytics run history
- Robust YOLO MAITE DataLoaders (#707)
- Project changelog and contributor guide

### Changed
- Moved XAITK detection baseline dataset into shared OD dataset module
- Consolidated analytics-store demo notebooks into a single tutorial
- Updated NRTK dependency to 1.0.3
- Updated dataeval dependency to 1.0.6
- Restored Poetry as the primary install path in getting-started docs
- Updated project license metadata and public documentation URLs
- Documented torchvision config file requirements, COCO metadata conventions, `index2label` mapping conventions, and IC NRTK warning filtering
- Migrated release publishing to PyPI Trusted Publishing / GitLab OIDC for production and TestPyPI API-token uploads for TestPyPI

### Fixed
- Capability metadata warning handling
- Deployment URL for GitLab Pages after team rename
- Scheduled Ray test stability
- Open figure warnings in tests and notebooks
- PyPI release workflow default-branch fetching

## [0.2.0] - 2026-05-28

### Added
- Ray backend for distributed job submission (#646)
- Markdown-to-PDF report export
- Manual PyPI publishing workflow
- GitHub Pages documentation mirror workflow

### Changed
- Jobs backend API renamed for clarity
- Deduplicated quick-start pages in the docs
- Moved analytics store guide into development docs

### Security
- Bumped Starlette lockfile to address BadHost CVE

### Fixed
- Removed unsupported extra from package metadata

## [0.1.8] - 2026-05-13

### Added
- Job-submission feature with Ray Serve backend for remote model inference
- Plugin system for extending capabilities
- Analytics Store records and extractors for XAITK explainability, NRTK robustness,
  dataeval shift, feasibility, and bias results
- Analytics Store end-user and developer documentation
- Sufficiency capability for image classification
- `FieldwiseDataset` protocol implementation
- `list_jobs` query parameters
- Python 3.12 to the test matrix
- `split_dataset` and `make_subset` dataset utility methods
- DataFrame input support for `Report.add_table`
- Dynamic versioning (hatch-vcs / Poetry)

### Changed
- Package renamed from legacy name to **checkmaite**
- Upgraded dataeval to 1.0.3
- Updated project license

### Removed
- PySpark removed from core dependencies (reduces install footprint)

### Fixed
- `print_serve_status` incorrectly reporting 0 healthy replicas
- Polars run-cache serialization error
- Flaky NRTK integration tests
- CI cache collision between dev and docs Poetry installs

## [0.1.7] - 2026-03-04

### Added
- DR compliance UI component
- Key concepts documentation page

### Changed
- Updated NRTK implementation for compatibility with NRTK 0.27.1
- Bumped Panel and Bokeh version pins

### Removed
- Git LFS (replaced with standard VCS storage)

## [0.1.6] - 2026-02-23

### Added
- Personas and workflows documentation

### Changed
- Upgraded dataeval dependency to 1.0

### Fixed
- Various bug fixes required for demo stability
- Resolved nightly CI failures

## [0.1.5] - 2026-02-17

### Added
- Analytics Store (initial implementation)
- `DataevalFeasibility` capability for object detection datasets

### Changed
- Docs re-organized for improved navigation

### Fixed
- NRTK capability cache bug

## [0.1.4] - 2026-01-28

### Added
- `FieldwiseDataset` protocol
- Cloud storage support for dataset loaders
- Nightly build pipeline for deprecated-support packages

### Changed
- `CapabilityRunner` renamed to `Capability`
- Upgraded xaitk-jatic to v0.8.0
- Codebase layout re-organized for clearer module boundaries
- Optimized COCO detection dataset metadata lookups
- Updated embedding computation method

### Removed
- Plugin architecture (superseded by simpler extension model)
- `Gradient` moved to optional dependency
- `RealLabel` and `Survivor` moved to optional dependencies

### Fixed
- NRTK fixes and related UI updates
- UI broken imports after architecture changes

## [0.1.3] - 2025-10-28

### Added
- Cached tasks as the replacement for `EvaluationTool`
- Input validation for dataeval drift and bias test stages
- `HEART` moved to optional dependency

### Changed
- Refactored Image Classification Test Stage API
- Upgraded to MAITE 0.8.2
- Upgraded dataeval

### Security
- Resolved multiple dependency security vulnerabilities

## [0.1.2] - 2025-06-18

### Added
- OD HEART adversarial robustness test stage
- NRTK Power User notebook
- Dataeval shift and feasibility tutorial notebooks

### Fixed
- Multi-model "add model" button in the UI

## [0.1.1] - 2025-06-13

### Added
- Dataeval bias tutorial notebook
- Dataeval cleaning tutorial notebook
- XAITK Power User notebook

### Fixed
- Real-data overnight test failures

## [0.1.0] - 2025-06-06

Initial public release of checkmaite.

---

[Unreleased]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.2.1...HEAD
[0.2.1]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.2.0...0.2.1
[0.2.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.8...0.2.0
[0.1.8]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.7...0.1.8
[0.1.7]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.6...0.1.7
[0.1.6]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.5...0.1.6
[0.1.5]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.4...0.1.5
[0.1.4]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.3...0.1.4
[0.1.3]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.2...0.1.3
[0.1.2]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.1...0.1.2
[0.1.1]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.1.0...0.1.1
[0.1.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/tags/0.1.0
