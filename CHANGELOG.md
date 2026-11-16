# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Evaluation tasks are no longer hard-coded into the core (#814). Each task is described by an `EvaluationTask` in the new `checkmaite.core.evaluation_tasks` registry, which gives its name, the namespace of its capabilities, how its predictions and targets are cached, and whether its inputs are one-shot streams. A package adds a task, such as semantic segmentation, by declaring an entry point in the `checkmaite.tasks` group, without changing CheckMAITE. Image classification, object detection and multi-object tracking are described the same way and register themselves when their namespaces are imported, with the aliases `ic`, `od` and `mot` used by DataMAITE. See the new "Adding an evaluation task" development guide.
- Predictions from a package-added task are cached; before, they failed validation against the built-in target types and were never cached.
- A task loaded from a `checkmaite.tasks` entry point must set `EvaluationTask.plugin_api_version`, checked against `PLUGIN_API_VERSION` like capability plugins. A missing or incompatible version, or a `target_adapter` that is not a complete pydantic `TypeAdapter`, is recorded as a failed entry point and does not affect other tasks.
- `EvaluationTask` takes keyword arguments only, is generic in its target type, and takes a pydantic `TypeAdapter` for one prediction or target, so any type pydantic validates works, including unions such as image classification's `ArrayLike`. Annotated with the task's target protocol, for example `EvaluationTask[SegmentationTarget]`, the type checker verifies that the adapter's type satisfies the protocol and that a task object passed as `task=` matches the model's target type.
- `Capability.version` (default `"1"`) and `Capability.cache_dependencies` are part of the run UID. The DataEval capabilities declare `dataeval`, NRTK Robustness `nrtk` and XAITK Explainable `xaitk-saliency`, and the exact installed version of each goes into the UID, so a library upgrade, including a patch release, no longer returns cached results computed with the old version. Bump `version` when a CheckMAITE change alters a capability's outputs. Each run records these versions in `run.versions`. MAITE Evaluation declares no library: MAITE only runs the user's model and metric, and the prediction cache it reads does not key on MAITE, so MAITE is treated as part of the user's environment, like PyTorch.
- A new analytics table, `run_versions`, records the library versions in `run.versions`, written automatically with each run: one row per library with `kind = 'library'`, the distribution `name` (for example `dataeval`) and its exact `version`. Join it on `run_uid` to tell results from before and after a library upgrade apart. The capability's own `version` only invalidates the cache and is not recorded. Runs of capabilities with no library dependencies, such as MAITE Evaluation, and runs written before this release have no rows. The `runs` table is unchanged.
- `checkmaite.core.component_checks.check_components` runs MAITE 0.10's runtime protocol checks (`maite.spotcheck`) on a model, dataset, metric and augmentation for a task, using a few batches from the dataset. Run it once when wiring up new components; it does not touch the cache. It needs the new `spotcheck` extra (`checkmaite[spotcheck]`), which installs `beartype` and requires `maite>=0.10,<0.11`.

### Changed
- **Breaking:** `cached_tasks.evaluate` and `predict` require `task=` naming a registered task; the task is never inferred from the data. An unknown task name raises `KeyError`, and predictions or targets that do not match the named task raise `ValueError`. The match is checked on the first non-empty batch of every fresh run, including with `use_cache=False` and dataloaders; before, a value that did not fit only disabled caching with a warning. Use `checkmaite.core.<task>.cached_tasks` for the same functions with the task bound.
- **Breaking:** every `Capability` subclass must set the class attribute `task` to a registered task, for example `task = "object_detection"`. Running one that does not raises `TypeError`, or `KeyError` for an unknown task.
- **Breaking for capability plugins:** a plugin capability must set `task`, and any `cached_tasks.evaluate` or `predict` call it makes must pass `task=`. `PLUGIN_API_VERSION` stays `1.0.0`, so a plugin written for 0.7.0 still loads but fails when it runs. Update plugins, including `checkmaite-plugins`, before upgrading.
- **Breaking:** container plans must name the task in `builtin:` capability references, such as `builtin:object_detection.MaiteEvaluation`. The short form `builtin:MaiteEvaluation`, which inferred the task from the plan's other objects, is a configuration error.
- `checkmaite.core.image_classification.cached_tasks`, `object_detection.cached_tasks` and `multiobject_tracking.cached_tasks` export `evaluate` and `predict` with the task bound, so they take the same arguments as `maite.tasks.evaluate` and `maite.tasks.predict`. They also re-export `evaluate_from_predictions`, which takes no task. `cached_tasks.for_task(task)` binds a package's own task the same way.
- The task is part of the prediction and metric-result cache keys, so an entry is never reused for a different task. The prediction cache validates predictions and targets with the task's target type, instead of a fixed union of the image-classification, object-detection and tracking types.
- Prediction and metric-result caches written by earlier versions are not reused. They are recomputed once, the first time each evaluation runs again.
- Every run UID changes once with this release, because the UID now includes the capability and library versions. Existing capability cache entries become misses and are recomputed on the next run. Analytics rows written by those reruns get new `run_uid` values, so rows from before and after the upgrade do not share run IDs.
- **Breaking:** capability configs reject unknown fields, so a removed option such as `target_metric_name`, or a misspelled one, raises a `ValidationError` instead of being ignored.
- **Breaking:** `CapabilityRunBase.compute_uid` takes a new required keyword argument, `versions`, such as `capability.cache_versions()`.
- The batch container resolves `builtin:<task>.Name` for every registered task, including aliases and tasks added by packages, instead of only `object_detection` and `image_classification`. `builtin:multiobject_tracking.MaiteEvaluation` now works.
- **Breaking:** the Pydantic models for cached object-detection and tracking targets moved from `checkmaite.core.cached_tasks` to `checkmaite.core.object_detection._task` and `checkmaite.core.multiobject_tracking._task`. Importing them from `checkmaite.core.cached_tasks` now fails.
- Upgraded DataEval from 1.0.6 to 1.2.0 and dataeval-plots from 0.0.6 to 0.0.16. CheckMAITE now uses `dataeval.data.View` and `dataeval.data.split_dataset` in place of the removed `dataeval.selection` and `dataeval.utils.data` APIs, and `image_stats` in place of the deprecated `compute_stats`. Some results change, because DataEval changed how it computes them:
    - Cleaning: brightness, darkness and the brightness percentiles are reported on the 0–255 display range for every image type, instead of 0–1. `DataevalCleaningRecord.mean_brightness` moves to the same scale. Contrast and sharpness do not change. The new scale does not change outlier flags on its own, because z-scores do not depend on scale, but the minimum population below can.
    - Cleaning: label statistics list every class in the dataset's `index2label`, including classes with no labels, so `class_count` can be larger than before. `class_imbalance_ratio`, `min_class_image_count` and `max_class_image_count` still use only the classes that appear.
    - Cleaning: z-score outliers are only judged for statistics with at least 20 values, so datasets with 11 to 19 images (or boxes) no longer report outliers.
    - Bias: balance scores are corrected for chance, so factors with many distinct values (such as per-image IDs or URLs) score lower. The classwise balance table no longer has a row for `class_label` against itself, and the factors table gains a `scored_as` column.
- Upgraded NRTK from 1.0.4 to 1.1.0. `NrtkRobustnessConfig.perturber_factory` is now typed as `nrtk.interfaces.PerturbFactory`, and the default factory is `nrtk.impls.perturb_factory.PerturberOneStepFactory`. NRTK 1.1 deprecates `PerturbImageFactory` and the `nrtk.impls.perturb_image_factory` classes; CheckMAITE does not support them, so configs must name the `nrtk.impls.perturb_factory` classes. NRTK 1.1 changes `PybsmPerturber` to convert simulated sensor values to pixels with a fixed, sensor-defined scale (`pixel_conversion_mode="radiometric"`) instead of stretching each image to 0–255, so robustness curves for pyBSM perturbers change; pass `"pixel_conversion_mode": "minmax"` in `perturber_kwargs` to get the NRTK 1.0 behaviour.
- **Breaking:** NRTK Robustness also evaluates the model on the unperturbed dataset and reports that baseline, so a separate MAITE Evaluation is no longer needed to read a robustness curve. `NrtkRobustnessOutputs` now describes the sweep itself: `perturber` (such as `"BrightnessPerturber"`), `theta_key`, `baseline` (the metric's results on the unperturbed dataset), `points` (one `NrtkRobustnessPoint` per perturbation level, with its `theta` value and the metric's results there) and `metric_key`. It replaces `perturbations`, a list of metric results that had to be matched to the perturber factory's thetas by position. Each run does one more evaluation, for the baseline. `NrtkRobustnessRecord` gains `baseline_value`, the score for the same metric on the unperturbed dataset, and the Markdown report shows the baseline on the plot and each level's change from it. A run with a factory that sweeps more than one parameter, such as `PerturberMultivariateFactory`, now fails before evaluating anything.
- **Breaking:** capabilities no longer read a metric's `return_key` attribute to decide which result to report. A capability that needs one number gets the key from its own config, as Lightning's `monitor` and Hugging Face's `metric_for_best_model` do; when the config does not name one and the metric returns exactly one number, that number is used. The key is part of the config, so it is part of the run UID. `checkmaite.core.metric_keys.resolve_metric_key` applies this rule and is available to plugin capabilities.
    - NRTK Robustness: new `NrtkRobustnessConfig.metric_key`. `extract()` skips results that are not single numbers, and `classes` and `*_per_class` results, such as an mAP metric's `map_per_class` and `classes`; before, they made it raise `TypeError`, or were recorded as metrics when torchmetrics returned them as single numbers. It is checked on the baseline results, so a missing or ambiguous key fails before the sweep instead of in the report. A metric that returns several numbers, and that previously relied on `return_key`, now needs `metric_key`.
    - Sufficiency: `DataevalSufficiencyConfig.target_metric_name` is replaced by `metric_key`, and the `"target_metric"` fallback is gone. The key is checked against the target metric's results on the first evaluation, where its value may be undefined (NaN); extra metrics are not considered.
    - MAITE Evaluation: new `MaiteEvaluationConfig.metric_keys`, mapping a metric ID to the result to highlight, for example `{"mAP_50": "map_50"}`. A metric with no entry has its only number highlighted, or none if it returns several; all results are still reported and recorded. Naming a metric that was not evaluated raises `ValueError`. `MaiteEvaluationRecord` gains `is_primary`, true on the overall record of the highlighted result, so the headline number of each run can be selected in SQL. For per-class results, the overall value is the one key left after the class values.
    - `return_key` remains a constructor argument of CheckMAITE's own metric wrappers, where it names or filters what `compute()` returns.
    - The CLEAR-MOT tracking metric (`clear_mot_metrics_factory()`) returns several numbers, so MAITE Evaluation no longer highlights MOTA unless `metric_keys` names it, for example `{metric.metadata["id"]: "mota"}`.
- NRTK Robustness includes the exact installed NRTK version in the augmentation ID it builds for each perturber, so cached predictions and metric results on perturbed images are not reused after any NRTK upgrade, including a patch release. Users of a shared cache with different NRTK versions no longer reuse each other's NRTK predictions.
- **Breaking:** CheckMAITE requires MAITE 0.10 (`maite>=0.10,<0.11`), up from `>=0.9.4,<0.10`; the lock resolves 0.10.3.
- Upgraded DataMAITE from 0.6.0 to 0.7.0, which fixes several dataset conversion bugs.
- Results cached before these upgrades are not reused: the DataEval and NRTK versions are part of the run UID, so the next run recomputes and gets a new `run_uid`. To migrate, re-run the capabilities you need and write them to the analytics store; the old rows are kept. Filter on the `run_versions` table (`kind = 'library'`, `name = 'dataeval'`, `version = '1.2.0'`) to compare results from the same version. See "After a library upgrade" in the key concepts guide, which also describes what the cache keys leave to you: keeping model, dataset and augmentation IDs accurate, and keeping one cache per Python environment.

### Fixed
- Tracking predictions, targets and evaluations from DataMAITE video datasets are now cached. Before, every `cached_tasks.predict` or `evaluate` call on them ran the model again and warned `Cache publication is disabled for this value`, because the cache could not store DataMAITE's lazy `frame_tracks` sequence or the `Fraction` `time_base` in each video's metadata. The cache now stores `frame_tracks` as a list of frames and stores `Fraction` values exactly. Cache files written by 0.7.0 still load.
- The Parquet analytics store reads a table's files with their columns combined, so a store written by an earlier version keeps working after a record gains a field, such as `baseline_value` or `is_primary` in this release. Before, the new column made writes skip deduplication and store duplicate rows, made `get_run_uri` report existing runs as missing, and left the new column out of `describe_table`. Errors reading existing files are no longer treated as an empty table, and each file is written under a temporary name and then renamed, so a crashed write cannot leave a truncated file.
- NRTK Robustness builds the augmentation ID from the perturber's `get_config()`, full class path and the NRTK and pyBSM versions. Before, pyBSM perturbers got a new ID on every run, because their simulator objects were hashed by memory address, so their predictions were never reused and the cache kept growing.
- NRTK Robustness copies each evaluation's metric results, so a metric that returns arrays or tensors sharing its internal state can no longer overwrite the baseline or earlier levels. Undefined (NaN or infinite) results are stored as `None`, so a cached run matches a fresh one; the Markdown report shows such levels as "undefined" and analytics records skip them. A metric whose only result is undefined on the baseline now fails, as one named with `metric_key` already did.

## [0.7.0] - 2026-11-02

### Added
- Python 3.14 support with `uv` and pip, with 3.14 added to the uv CI test matrix (SR-4-G-1, #799). conda stays at 3.11–3.13: conda-forge `nrtk` 1.0.4 requires `simsimd <7`, and conda-forge only builds `simsimd` 7.x for Python 3.14. The containers stay on Python 3.12; their private runtime lock is re-locked for the wider Python range so it keeps matching the root lock.
- `ROADMAP.md` with planned features and releases for CheckMAITE and the CheckMAITE UI through April 2027, also published as the Roadmap page of the docs (DR-2.2-H-3, #799).
- A short, runnable code example in the README and Get started page, install instructions for pip, uv, Poetry, and conda from PyPI and from source, and the product tagline and needs statement on the README and docs home page (DR-2.2-H-1, #799).

### Removed
- Python 3.10 support. `requires-python` is now `>=3.11, <3.15`, and the 3.10 CI legs are gone. The fixed Pillow release requires Python 3.11, so 3.10 installs could not be kept free of its Critical and High CVEs (DSOR-2-H-2, #799).
- `conda-lock` is no longer in the `dev` dependency group. It caps `dulwich` below 0.25 and `virtualenv` below 21 (and so `filelock` below 4), which held all three on vulnerable releases. Run it with `uvx --from "conda-lock>=3,<5" conda-lock ...`, as described in the development setup guide; the pre-commit hook already installs it separately (#799).
- VisDrone model support is no longer in the base install. Install the new `visdrone` extra (`checkmaite[visdrone]`), which forwards to `modelmaite[visdrone]`. `httpx`, `smqtk-detection` and `numba` are no longer core dependencies; checkmaite does not import them, and they only served modelmaite's VisDrone wrapper. VisDrone and ONNX tests skip when their extra is not installed.
- `transformers[timm]` is no longer a dependency. Nothing in checkmaite has imported it since the UI was removed in 0.4.0, and dropping it also removes `timm`, `huggingface-hub`, `tokenizers`, `safetensors` and their dependencies from the lock.

### Changed
- Replaced `xaitk-jatic` with `xaitk-saliency[maite]>=0.16.0`, which now ships the same MAITE interop (`MAITEImageClassifier`, `sal_on_dets`); `xaitk-jatic` is being deprecated upstream. This lifts the last NumPy < 2 caps on Python 3.11 and 3.12 (with `pybsm` 0.16.1), so every supported Python now installs NumPy 2.
- Release containers are pushed to Harbor with OCI media types instead of Docker media types (CS-1-H-1). BuildKit pushes the same cached build that was tested and scanned; the release job checks the pushed manifest's media type and config digest before signing. Tags 0.6.1 and earlier cannot be republished with the new release jobs.
- Pipelines include the program's `cs-compliance` component, which checks the latest published Harbor containers (`cs_images`).
- CI no longer installs `checkmaite-plugins`. It pins `numpy<2` and Python `<3.12` through unmaintained upstreams (`gradient`, `survivor`, `reallabel`, `heart-library`) that hold CVE-affected versions, so the `gradient` report tests now skip in CheckMAITE CI ([checkmaite-plugins#2](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins/-/work_items/2)). Real plugin discovery is now tested with a throwaway entry-point distribution; HEART-specific checks belong in `checkmaite-plugins`.
- The plugins install docs now add `setuptools<82` and `pybsm<0.16.1`: `gradient` imports `pkg_resources`, which setuptools 82 removed, and `checkmaite-plugins` does not cap it yet ([checkmaite-plugins#2](https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite-plugins/-/work_items/2)).
- Added `pyarrow` to the `dev` dependency group, so the Parquet cache codec tests run instead of skipping.
- The `onnx` and `onnx-cuda` extras no longer cap `onnxruntime` below 1.24, a cap that only existed because 1.24 has no Python 3.10 wheels.
- The `onnx` and `onnx-cuda` extras now forward to `modelmaite[onnx]` and `modelmaite[onnx-cuda]`, so modelmaite sets the ONNX version bounds. modelmaite 0.2.0 caps `onnxruntime` below 1.29, so the lock moves `onnxruntime` and `onnxruntime-gpu` from 1.31.1 to 1.28.0 (no known CVEs), and Python 3.13 resolves `onnx` 1.23.2 because modelmaite excludes 1.22.0 there.
- `load_models` (object detection and image classification) names the `checkmaite[visdrone]` or `checkmaite[onnx]` extra when a VisDrone or ONNX dependency is missing.

### Fixed
- `create_pdf_output` passes the PDF's path to `xhtml2pdf`, so images still embed when the output directory is outside the working directory. Newer `xhtml2pdf` releases only read local files under the document's directory, which defaults to the working directory.

### Security
- Container scans also run Grype, the program's scanner, alongside Trivy (CS-2-H-2), and a Trivy gate on Medium findings (DSOR-3-H-2). Both report without blocking until findings with no available fix are remediated or excepted; `CHECKMAITE_STRICT_SCAN_GATES=true` makes them block. Trivy High and Critical findings still block, and Grype reports are kept with the Trivy ones.
- Removed the twenty "Accepted at Maturity S" scan acceptances; the exceptions file now accepts only approved program exceptions, and the Grype gate accepts none.
- Raised the minimum `pillow` to 13.0.0 and the minimum `xhtml2pdf` (`reporting` extra) to 0.2.19, which fixes an OS command injection (High). Raised the `dev` group's minimum `virtualenv` to 21.14.9 (#799).
- Upgraded locked dependencies with dependency-scan findings (DSOR-2-H-2, #799): `pyjwt` 2.15.1, `nbconvert` 7.17.2, `pypdf` 6.21.4, `tornado` 6.6, `oauthlib` 4.0.0, `xhtml2pdf` 0.2.28, `virtualenv` 21.15.1, `filelock` 4.2.4, `msgpack` 1.2.4, and `opentelemetry` 1.45.1. `dulwich` and `gitpython` are no longer locked.
- Upgraded the container's bundled-ffmpeg packages (CS-2-H-2, #799): `av` 19.0.2 on Python 3.12 and later, and `opencv-python-headless` 5.0.0.93 on every Python version, removing the Critical ffmpeg finding in 4.11 from the 3.12 batch container. Python 3.11 stays on `av` 18.1.0, since PyAV 19 has no 3.11 wheels.
- `setuptools` moves from 81.0.0 to 84.2.1, fixing its known CVEs, now that `xaitk-saliency` 0.16.0 no longer caps it below 82 (#584).
- Upgraded `pyasn1` to 0.6.5, fixing four High CVEs in 0.6.4 that the container scan reports (CS-2-H-2, #799).

## [0.6.1] - 2026-10-22

### Changed
- Container builds in merge request and `main` pipelines, including the daily scheduled pipeline, now fail on the same scan gate as a release instead of only reporting findings, so a newly published finding is caught before it blocks a release.
- The container scan gate fails only on High and Critical findings. Medium findings are still reported, but no longer block publication while CheckMAITE targets Maturity S, where DSOR-3 is not required.

### Security
- Upgraded `pillow` to 13.0.0 on Python 3.11 and later, fixing 28 Medium, High, and Critical CVEs in Pillow 12.3.0 (3 Critical). Pillow 13 requires Python 3.11, so Python 3.10 installs remain on 12.3.0 and are still affected.
- Upgraded `certifi` to 2026.10.19, removing the emSign root CAs whose intermediate CA keys were compromised (GHSA-6q48-6c7w-9gw2).
- These findings blocked publishing the 0.6.0 containers to Harbor, so 0.6.1 is the first release with containers since 0.5.0. The 0.6.0 Python package does not pin either package, so new installs of it on Python 3.11 and later already get the fixed versions.

## [0.6.0] - 2026-10-22

### Added
- `checkmaite.core.multiobject_tracking` namespace with a `MaiteEvaluation` capability for MAITE multi-object-tracking (MOT) models and datasets, a `load_mot_dataset` loader for MOTChallenge datasets and flat MP4 folders that emits every frame of every video and gives each dataset a cache id derived from its root, format and options, and the class-agnostic CLEAR-MOT/identity metric approved in !611 (`clear_mot_metrics_factory`: MOTA, MOTP, IDF1, identity switches), computed with `motmetrics`. That work was merged into the #718 feature branch but left out when the branch was rebased onto `main` (#723).

### Changed
- `MaiteEvaluation` treats a metric value that is not a finite number (NaN or infinity) as undefined: it is left out of the scores and shown as `null` in the run, and the evaluation fails with an error if it is the headline value. Previously such values were written to the cache as `null`, so a cached rerun lost them or failed.
- Container builds retry failed Ubuntu snapshot downloads (`Acquire::Retries=5`), so a short `snapshot.ubuntu.com` outage no longer fails the build.

### Fixed
- Locked `polars` 1.44.2 instead of 1.43.0, which was yanked from PyPI.

## [0.5.0] - 2026-10-20

### Added
- A batch container that runs one YAML run plan with the `checkmaite-container` command, writes durable results and exits, with a JSON Schema for the run plan. CPU and NVIDIA CUDA variants for Linux AMD64, built on digest-pinned Ubuntu 24.04 bases with operating-system packages fixed to one archive snapshot, running as a non-root user. Python warnings are logged with the run.
- Release pipelines publish the CPU and CUDA containers to Harbor (`openteams/checkmaite/{cpu,cuda}`), signed with Cosign through the program AWS KMS key, with SPDX SBOM and Trivy vulnerability attestations. Publication is blocked on any Medium, High or Critical finding not listed in `docker/ci/dsor3-exceptions.trivyignore.yaml`.
- The docs home page maps T&E tasks to the CheckMAITE features that support them (#803).
- CI runs GitLab secret detection, exports a CycloneDX SBOM artifact and checks Markdown with markdownlint; every test is marked `required` or `optional` (#767, #770, #782, #764).
- Every built-in capability's Markdown report ends with a Provenance section: run ID, capability ID, CheckMAITE version, the dataset, model and metric IDs, and the run configuration. A saved or shared report now identifies exactly what produced it without a separate metadata file (IR-3-S-3, #758).
- CI runs the DevSecOps `validate-pipeline` and `pipeline-report` components, producing the program's automated standards report (pipeline summary, program standards compliance and Grype scan reports) as a pipeline artifact and GitLab Pages report for product reviews (DR-1.3).
- System requirements reference page covering recommended hardware, supported architectures, and internet access for the Python library (#781).
- Documented critical limitations and prerequisites in the README and docs home page (#773). Conventions now list VisDrone detection and describe object-detection boxes as `float32` `xyxy`.
- Added the CDAO funding acknowledgment to the README and docs home page (#774).
- Glossary of key CheckMAITE, MAITE, and JATIC tool terms under Reference (#772).
- Step-by-step how-to for saving capability runs to the analytics store and querying them (#772).
- Concept explanations for saliency and distribution shift (drift and out-of-distribution detection) under Explanations, aligned with the xaitk-saliency and DataEval explanations (#780).
- `CONTRIBUTING.md`, `SECURITY.md`, a GitLab bug report issue template, and GitHub issue forms for bug reports and feature requests, documenting how to report bugs, request features, and report vulnerabilities, and how issues are managed (#769).
- A MAITE Evaluation tutorial covering baseline model evaluation with several metrics, per-class results, and object-detection postprocessing (#777).
- Documented how to override an ONNX model's input height, width, and batch size from the container run plan (IR-3.1-S-4), with a container test and commented example in `docker/example-run.yaml`.
- Python 3.13 support, with 3.13 added to the CI test matrix (#802). uv and pip support 3.10–3.13; conda supports 3.11–3.13, and the committed conda lock now resolves Python 3.13. The conda CI legs lock each matrix version and fail if the env is on a different one.
- MP4 video datasets load on a default install: flat folders of H.264 and MPEG-2 `.mp4` videos load through DataMAITE as MAITE MOT datasets that `cached_tasks.predict` accepts, with a how-to (IR-3.3-S-1) (#794).

### Changed
- Bumped `datamaite` from 0.5.0 to 0.6.0 and now install it with its `fmv`, `od` and `ic` extras; `fmv` adds PyAV. conda installs DataMAITE from conda-forge instead of PyPI, with PyAV listed separately because conda packages have no extras (#794).
- The README and docs home page no longer say there is no web UI: they name the separate CheckMAITE web application (`checkmaite-frontend`) and the batch container alongside the Python API.
- The manual release pipeline now creates the GitLab Release, with notes from `CHANGELOG.md` and links to the PyPI package, after `publish-pypi` succeeds.
- Reorganized the docs nav into Get started, Tutorials, How-to guides, Explanations, and Reference, without moving any pages, and cross-linked every tutorial with its related how-to guides (#772, #778).
- How-to guides, tutorials, and the dataset conventions now cite reported user problems and their fixes or limitations: TLS-proxy weight downloads, notebook-crash job recovery, durable report artifacts, YOLO labels outside `labels/`, non-deterministic DataEval loaders, and multivariate NRTK runs (#779).
- Framed each tutorial by persona and T&E workflow stage, and added the staged workflow to the Personas reference (#776).
- conda no longer claims Python 3.10 support: conda-forge stopped building `ray-core` for 3.10 after 2.52.1, below checkmaite's `ray-core >=2.55`. The former conda 3.10 CI leg was silently running 3.11. uv and pip still support 3.10 (#802).
- Bumped `modelmaite` from 0.1.0 to 0.2.0, the first release that supports Python 3.13 and numpy 2; 0.1.0 requires Python `<3.13` and numpy `<2` (#802).
- Python 3.13 installs numpy 2.x (with numba 0.68 and llvmlite 0.50), while 3.10–3.12 stay on numpy 1.26. `pybsm` (via `nrtk[pybsm]`), `smqtk-classifier`, and `xaitk-jatic` still cap numpy below 2 on Python <3.13 (#802).

### Security
- Release container scans accept twenty Medium findings (nineteen unfixed Ubuntu 24.04 CVEs, CVE-2026-59890 in setuptools) without a program exception while CheckMAITE targets Maturity S; entries expire 2027-01-05.
- Upgraded `aiohttp` to 3.14.5 and `multidict` to 6.9.1 (with `yarl` 1.25.1), fixing CVE-2026-111064, GHSA-2g87-pp6c-29x5 and CVE-2026-104874 in the container images.

## [0.4.0] - 2026-09-24

### Added
- `maite.protocols.*` and `maite.tasks` entry points for the MAITE components CheckMAITE contains: metrics, YOLO DataLoaders, the XAI explainable-detection baseline dataset, the Ray Serve object-detection and image-classification clients, and the cached `predict` / `evaluate` / `evaluate_from_predictions` tasks. Model wrappers are advertised by modelmaite and the on-disk dataset classes by datamaite.
- Dependency on `datamaite` 0.5.0 for native MAITE-compatible dataset loading (#717)
- Native MAITE fieldwise access (`get_input`/`get_target`/`get_metadata`), full COCO `images[]` datum-metadata fields, per-box VisDrone truncation/occlusion metadata, YOLO `split`/`yaml_file`/`ann_dir` loader options, and recursive YOLO image-classification discovery, all provided by datamaite 0.5.0 with no checkmaite-side adapters (#717)
- Remote (fsspec/UPath) dataset roots for COCO, YOLO, and VisDrone still-image datasets, with `storage_options` on the loader factories. A configured `UPath` root or override is passed through as an object, so its own filesystem options (credentials, endpoints) are kept. A local `ann_file`/`ann_dir` override stays local under a remote root; a local YOLO `ann_dir` under a remote root raises `ValueError` on datamaite 0.5.0 instead of loading images with no labels (#717)
- `DatasetSourceError`: with `strict_annotations=True` (the default), recognized datamaite row-rejection warnings -- skipped or dropped records, unreadable annotation files, and annotations stripped of their label -- fail dataset loading with aggregated `file:line` diagnostics. This is a best-effort guard that matches datamaite's log text; it does not currently guarantee complete source integrity. `strict_annotations=False` restores best-effort loading and reports the same diagnostics as a warning (#717)
- Multi-metric image-classification and object-detection evaluation through one shared MAITE inference pass, including when caching is disabled.
- Attributed `MaiteEvaluationMetricError` failures for metric reset, update, compute, and result normalization.
- Ray job scheduling status, scheduling deadlines, worker placement diagnostics, bounded job labels, and per-scope admission limits.
- `JobSubmissionError`, exported from `checkmaite.jobs`, which the Ray job backend raises with the failed submission phase and, when known, the job ID if a job fails before its handle is returned.
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
- The deprecated Gradient table-of-contents slides for DataEval bias, cleaning, and sufficiency reports now show a short text column instead of a decorative image, and the packaged `checkmaite/assets` image directory has been removed.
- Release publishing now pins the build backend (`hatchling` 1.31.0, `uv-dynamic-versioning` 0.14.0), bounds the upload wait, skips artifacts already accepted when a publish job is retried, and verifies every published artifact's digest against the locally built artifact.

### Removed
- **Breaking:** Removed the Panel-based web UI (`checkmaite.ui`), the `checkmaite-serve` console script, and the `ui` optional-dependency extra (`holoviews`, `jupyterlab`, `nbconvert`, `panel`, `streamz`), along with the UI tests, the UI-driven end-to-end tests, and the interactive UI documentation. Use the Python API to configure and run capabilities.
- Removed the Ray backend's `registry_actor_name` option. New clients use one fixed registry actor per Ray namespace and do not discover registries created with the previous scope-hashed names. Before upgrading, finish or cancel in-flight jobs with the previous CheckMAITE release, or keep that client available until the Ray cluster is recycled.
- CheckMAITE's in-tree model wrapper implementations and their ONNX/torchvision helper utilities in `checkmaite.core._utils`, now maintained in modelmaite (#718)
- CheckMAITE's `YoloClassificationDataset`, `CocoDetectionDataset`, `YoloDetectionDataset`, and `VisdroneDetectionDataset` objects and their concrete Torch-output contract (#717)
- CheckMAITE's own `DetectionTarget` dataclass. Dataset code, the XAITK prediction-backed dataset, and the Ray Serve client now use `modelmaite.object_detection.DetectionTarget`, the same type the modelmaite model wrappers return (#717)

### Fixed
- Ray jobs now become `RUNNING` only after their capability worker begins execution, and terminal controllers can clean themselves up on quiet clusters.
- Passed datum metadata to metrics when evaluating cached predictions.
- Restored cache loading for Torch tensor subclasses on Torch 2.6 and later and for PIL images backed by temporary buffers.
- Preserved generator-backed MOT frames in fresh augmented-data debugging responses.
- `DataevalCleaningRun.collect_md_report()` now writes its temporary histogram plots to a per-run directory and removes them after embedding, so concurrent Markdown report runs no longer overwrite each other's plots.

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

[Unreleased]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.7.0...HEAD
[0.7.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.6.1...0.7.0
[0.6.1]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.6.0...0.6.1
[0.6.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.5.0...0.6.0
[0.5.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.4.0...0.5.0
[0.4.0]: https://gitlab.jatic.net/jatic/orchestration-interoperability/checkmaite/-/compare/0.3.0...0.4.0
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
