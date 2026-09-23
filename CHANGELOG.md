# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Dependency on `datamaite` 0.5.0 for native MAITE-compatible dataset loading (#717)
- Native MAITE fieldwise access (`get_input`/`get_target`/`get_metadata`), full COCO `images[]` datum-metadata fields, per-box VisDrone truncation/occlusion metadata, YOLO `split`/`yaml_file`/`ann_dir` loader options, and recursive YOLO image-classification discovery, all provided by datamaite 0.5.0 with no checkmaite-side adapters (#717)
- Remote (fsspec/UPath) dataset roots for COCO, YOLO, and VisDrone still-image datasets, with `storage_options` on the loader factories. A configured `UPath` root or override is passed through as an object, so its own filesystem options (credentials, endpoints) are kept. A local `ann_file`/`ann_dir` override stays local under a remote root; a local YOLO `ann_dir` under a remote root raises `ValueError` on datamaite 0.5.0 instead of loading images with no labels (#717)
- `DatasetSourceError`: with `strict_annotations=True` (the default), recognized datamaite row-rejection warnings -- skipped or dropped records, unreadable annotation files, and annotations stripped of their label -- fail dataset loading with aggregated `file:line` diagnostics. This is a best-effort guard that matches datamaite's log text; it does not currently guarantee complete source integrity. `strict_annotations=False` restores best-effort loading and reports the same diagnostics as a warning (#717)
- Multi-metric image-classification and object-detection evaluation through one shared MAITE inference pass, including when caching is disabled.
- Attributed `MaiteEvaluationMetricError` failures for metric reset, update, compute, and result normalization.
- Ray job scheduling status, scheduling deadlines, worker placement diagnostics, bounded job labels, and per-scope admission limits.
- Independent `artifact_store` configuration for Ray job backends, providing durable local, S3, GCS, or Azure storage for oversized inline reports without coupling report storage to the analytics store.
- Cache schema version 1 for serialized Pydantic cache entries.
- A strict, lossless cache validation option alongside the more flexible default serialization.
- Support for extension fields in cached MAITE datum metadata.
- MOT prediction, target, and metadata caching in flexible serialization mode, including `track_ids`.
- Dependency on `modelmaite` 0.1.0 for the MAITE model wrappers (#718)

### Changed
- `JobStatus` now includes `SCHEDULING` for distributed jobs waiting on worker resources.
- The public `Job` protocol now requires a `job_name` property.
- Non-numeric Ray task resource quantities now raise `TypeError` before job registration.
- Dataset factories now return datamaite `ObjectDetectionDataset` and `ImageClassificationDataset` objects directly; images and targets use datamaite's NumPy MAITE surface rather than a second Torch-backed checkmaite representation (#717)
- Dataset configuration dispatches by wire format (`coco`, `yolo`, `visdrone`) rather than by CheckMAITE implementation-class name (#717)
- Image-classification datum ids are root-relative and include the split (`test/cat/cat.jpg`); `MissingYoloDataSplitError` now lists the splits present under the root and covers empty splits (#717)
- Loader provenance metadata (`source_line`, `yolo_bbox`, `label_file`, `annotation_file`, `source_file_name`, `source_format`, `variant`) is excluded from DataEval bias factors by default through `DataevalBiasConfig.metadata_to_exclude`; real annotation attributes such as VisDrone truncation and occlusion remain per-object factors (#717)
- Dataset loading uses datamaite's lazy OpenCV image decoding and image-driven VisDrone discovery (#717)
- `MaiteEvaluation` now accepts one or more metrics, canonicalizes them by metadata ID, and returns results under `outputs.metrics[metric_id]`. The previous single-metric output attributes have been removed. Its run-cache identity now includes a schema version so incompatible single-metric run entries are not loaded as nested outputs.
- Ray job clients now use one fixed internal registry actor per Ray namespace while retaining idempotency scopes as logical lookup and deduplication partitions; select another namespace for an independent registry.
- Existing Ray registry actors now complete a version and immutable-configuration handshake before clients reuse them.
- **Breaking:** Ray job backends now require `artifact_store` during configuration so missing or unsupported report storage is rejected before any capability runs or analytics are committed. Process-local `memory` stores and relative local paths are rejected; local paths must be absolute and shared by the client and every Ray node.
- Ray workers now externalize oversized inline reports under job- and run-scoped, path-safe content-addressed keys that deduplicate retries. Mismatched objects are replaced through a supported atomic local rename or object-store commit; if safe replacement is unavailable or fails, publication fails without deleting the existing key. Report producers remain responsible for returning self-contained inline content or durable `ArtifactReport` URIs; job backends do not attempt to discover or rewrite embedded file references.
- The 256 KiB inline-report limit is now enforced when constructing job-result metadata, allowing report producers to generate a larger self-contained `InlineTextReport` for the backend to externalize.
- Built-in Markdown reports now embed generated plots and saliency images instead of returning worker-local image paths.
- Raised the minimum Pydantic version to 2.12.0 and typing-extensions to 4.14.1 for PEP 728 TypedDict support.
- Cache entries and binary files now use failure-safe publication and clean up partial writes.
- Binary cache references are decoded only while loading cache entries.
- Made `evaluate()` the fundamental cached task, with `predict()` as its `metric=None` convenience wrapper.
- Required `metadata_batches` for `evaluate_from_predictions()` and an explicit `inference_id` for result caching.
- Restricted `return_augmented_data` to the Boolean MAITE API; full-data requests now run fresh and publish nothing.
- Changed prediction and evaluation cache identities, so entries created by earlier versions will be cold.
- Upgraded MAITE from 0.9.2 to a `>=0.9.4,<0.10` range. The floor is 0.9.4, the release that introduced the native multi-object tracking protocols CheckMAITE imports unconditionally, and a range rather than an exact pin avoids lockstep bumps with checkmaite-plugins' circular test dependency. The resolved version is 0.9.5.
- Relaxed the IPython dependency upper bound so Python 3.11+ can use IPython 9 while Python 3.10 resolves a compatible 8.x release.
- Upgraded nrtk to 1.0.4.
- Lowered declared floors for `torchmetrics` (1.0.0), `scikit-learn` (1.5.2), `matplotlib` (3.7.1), and `pytest` (7.3.1) to the SR-4-H-2 program table. Dropped `extended_summary` and `average` from object-detection mAP factory kwargs so the torchmetrics 1.0.0 constructor is usable. Metric cache identities that hash those kwargs will change.
- Documented the supported OS, Python (uv vs conda), and GPU baseline in the README and install guide, and pointed clone, contributing, and docs URLs at this project.

- Model wrappers (`TorchvisionODModel`, `VisdroneODModel`, `OnnxODModel`, `TorchvisionICModel`, `OnnxICModel`) are now modelmaite's, re-exported from their historical CheckMAITE import paths; `ModelSpecification`, `load_models`, and `SUPPORTED_MODELS` remain CheckMAITE's config-facing contract (#718)
- `load_models` now delegates dispatch to modelmaite's native factories (IC re-exported directly; OD translates the legacy VisDrone `model_weights_path` key to `model_pickle_dir` first). Keyword arguments now reach VisDrone wrappers, a missing or unsupported `model_type` raises modelmaite's `ValueError` (previously `KeyError`/`RuntimeError`), and unsupported-type errors list the supported models (#718)
- Model prediction targets are NumPy-backed (modelmaite) rather than Torch tensors; the VisDrone wrapper name dropped the doubled `centernet-` prefix; missing optional dependencies and failed weight downloads now raise modelmaite's stricter, hint-bearing errors (#718)
- **Breaking:** the ONNX wrappers now accept only `uint8` integer images. CheckMAITE's removed `_normalize_image` accepted any non-negative integer dtype and scaled by that dtype's maximum; modelmaite 0.1.0 raises `TypeError` for any integer dtype other than `uint8`, so MAITE datasets yielding `uint16` or positive `int16` images must convert to `float32` in `[0, 1]` before calling the wrapper. Parity is restored upstream in modelmaite !15 and will return here with the pin bump to the release carrying it (#718)

### Removed
- Removed the Ray backend's `registry_actor_name` option. New clients use one fixed registry actor per Ray namespace and do not discover registries created with the previous scope-hashed names. Before upgrading, finish or cancel in-flight jobs with the previous CheckMAITE release, or keep that client available until the Ray cluster is recycled.
- CheckMAITE's in-tree model wrapper implementations and their ONNX/torchvision helper utilities in `checkmaite.core._utils`, now maintained in modelmaite (#718)
- CheckMAITE's `YoloClassificationDataset`, `CocoDetectionDataset`, `YoloDetectionDataset`, and `VisdroneDetectionDataset` objects and their concrete Torch-output contract (#717)
- CheckMAITE's own `DetectionTarget` dataclass. Dataset code, the XAITK prediction-backed dataset, and the Ray Serve client now use `modelmaite.object_detection.DetectionTarget`, the same type the modelmaite model wrappers return (#717)

### Fixed
- Ray jobs now become `RUNNING` only after their capability worker begins execution, and terminal controllers can clean themselves up on quiet clusters.
- Passed datum metadata to metrics when evaluating cached predictions.
- Restored cache loading for Torch tensor subclasses on Torch 2.6 and later and for PIL images backed by temporary buffers.
- Preserved generator-backed MOT frames in fresh augmented-data debugging responses.

## [0.3.0] - 2026-07-24

### Added
- Configurable MAITE evaluation inference batch size
- Optional CPU object-detection evaluation postprocessing, disabled by default, with confidence filtering, NMS, class-agnostic NMS, and maximum-detection settings
- Typed inline-text and artifact capability report models

### Changed
- Replaced Poetry with uv and Hatchling for dependency management, development, CI, and packaging; development and documentation dependencies now use PEP 735 dependency groups
- Prediction and evaluation caches now distinguish inference batch sizes, so existing cached evaluations are cold after upgrading
- Capability reports now use typed inline or artifact models instead of plain strings, and job results expose an optional typed `report` instead of the untyped `summary` mapping
- Capability extensions must now return `InlineTextReport` or `ArtifactReport` from `collect_md_report()`; returning a plain string is no longer accepted and causes submitted jobs to fail during result construction
- Inline report content is limited to 256 KiB; oversized reports return a small placeholder so completed analytics remain accessible, while large, binary, or multi-file reports should use durable artifact references
- Detached Ray registry or controller actors holding pre-0.3.0 completed job records cannot reattach those results; restart detached actors when upgrading

### Fixed
- JSON Schema generation for torch `Device` fields

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

Initial public release of CheckMAITE.

---

[Unreleased]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.3.0...HEAD
[0.3.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.2.2...0.3.0
[0.2.2]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.2.1...0.2.2
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
