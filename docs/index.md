# CheckMAITE Documentation

CheckMAITE is a Python API which makes testing and evaluation of models and
datasets straightforward and reproducible. It can be used to run a wide variety
of **Model Evaluation** and **Dataset Analysis** investigations for both
**Object Detection** and **Image Classification** computer vision problems.

It was built as an integration point for all [CDAO
JATIC](https://cdao.pages.jatic.net/public/) tools and has since expanded focus.
Our goal is to make T&E easier for analysts!

## Supported T&E tasks

Each task maps to the CheckMAITE feature that performs it. Model wrappers and
dataset loaders come from the `models` and `dataset_loaders` modules of
`checkmaite.core.object_detection` (OD) and
`checkmaite.core.image_classification` (IC). The dataset and model evaluation
capabilities are available from both of those packages. Other features show
their full import path.

| Workflow stage | T&E task | Feature | Docs |
| --- | --- | --- | --- |
| Bring your model and dataset | Wrap an OD model | `TorchvisionODModel`, `VisdroneODModel`, `OnnxODModel` | [Object Detection Workflow via API](get-started/checkmaite_api_od.ipynb), [Wrap an ONNX object detection model](tool-usage/onnx_object_detection_wrapper.ipynb), [ONNX model wrappers](development/onnx_model_wrappers.md) |
| Bring your model and dataset | Wrap an IC model | `TorchvisionICModel`, `OnnxICModel` | [Image Classification Workflow via API](get-started/checkmaite_api_ic.ipynb), [ONNX model wrappers](development/onnx_model_wrappers.md) |
| Bring your model and dataset | Load an OD dataset (COCO, YOLO, VisDrone) | `load_coco_detection_dataset`, `load_yolo_detection_dataset`, `load_visdrone_detection_dataset` | [Object Detection Workflow via API](get-started/checkmaite_api_od.ipynb) |
| Bring your model and dataset | Load an IC dataset (YOLO) | `load_yolo_classification_dataset` | [Image Classification Workflow via API](get-started/checkmaite_api_ic.ipynb) |
| Curate the dataset | Find low-quality and duplicate data | `DataevalCleaning` | [Cleaning](tool-usage/dataeval_linting_tutorial.ipynb) |
| Curate the dataset | Detect bias and correlations | `DataevalBias` | [Bias](tool-usage/dataeval_bias_tutorial.ipynb) |
| Curate the dataset | Compare with operational data (drift and OOD) | `DataevalShift` | [Shift](tool-usage/dataeval_shift_tutorial.ipynb) |
| Curate the dataset | Check that the task is achievable | `DataevalFeasibility` | [Feasibility](tool-usage/dataeval_feasibility_tutorial.ipynb) |
| Evaluate the model | Measure baseline performance | `MaiteEvaluation` | [MAITE Evaluation](tool-usage/maite_evaluation_tutorial.ipynb) |
| Evaluate the model | Test robustness to realistic perturbations | `NrtkRobustness` | [NRTK](tool-usage/nrtk_tutorial.ipynb) |
| Evaluate the model | Explain what drives predictions | `XaitkExplainable` | [XAITK](tool-usage/xaitk_tutorial.ipynb) |
| Record and compare results | Store and query results across runs | `checkmaite.core.analytics_store.AnalyticsStore` | [Analytics Store](tool-usage/analytics_store_tutorial.ipynb) |
| Run at scale | Run capabilities on a Ray cluster | `checkmaite.jobs.RaySimpleJobBackend`, `checkmaite.jobs.RayJobBackend` | [Ray Simple Job Submission](tool-usage/ray_simple_job_submission_tutorial.ipynb), [Ray Job Submission](tool-usage/ray_job_submission_tutorial.ipynb) |
| Run at scale | Serve models remotely | `checkmaite.core.serving.rayserve`: `RayObjectDetectionDeployment`, `RayImageClassificationDeployment`, `RayObjectDetectionClient`, `RayImageClassificationClient` | [Run remote inference with Ray Serve](get-started/checkmaite_ray_serve.ipynb) |

See the [T&E workflow](reference/personas.md#te-workflow) for how the stages fit
together and who leads each one.

## Limitations and prerequisites

--8<-- "README.md:limitations"

<!-- markdownlint-disable MD053 -- used by the README snippet included above -->
[limitations-conventions]: reference/conventions.md
[limitations-supported-environment]: get-started/install_setup.md#supported-environment
<!-- markdownlint-enable MD053 -->

## Who is CheckMAITE for?

* **No prior knowledge of specific JATIC tools** - Our users may or may not be
  familiar with the individual [JATIC
  products](https://cdao.pages.jatic.net/public/products/). They are familiar
  with their datasets and models and simply want to perform testing without
  needing to learn a new tool (or suite of tools).
* **Have their own models and datasets** - Our users are expected to come with
  arbitrary models and datasets. Among other out-of-the-box models, we enable
  loading default weights from torchvision while also providing a framework for
  users to bring their own.

## Quick start

New to CheckMAITE? Start with the [Get started guide](get-started/index.md) for
setup, API workflows, remote inference, and saving or querying results.

## About the JATIC program and CDAO

The Joint AI Test Infrastructure Capability (JATIC) program develops software
products for AI Test & Evaluation (T&E) and AI Assurance. The program is managed
by the Assessment & Assurance Division of the [DoD Chief Digital and Artificial
Intelligence Office (CDAO)](https://www.ai.mil/). It is funded from FY23-FY29.

!!! type "Program Mission"

    Develop software to accelerate and enable AI model test and evaluation for
    testers across the Department of Defense (DoD) enterprise, including DoD
    programs, research laboratories, industry partners, and academia in order to
    provide insight on the performance, effectiveness, robustness, and safety of
    the DoD's AI-enabled systems.

Learn more in the [CDAO JATIC program documentation](https://cdao.pages.jatic.net/public/).

### CDAO Funding Acknowledgment

--8<-- "README.md:acknowledgment"
