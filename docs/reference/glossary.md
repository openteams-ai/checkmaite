# Glossary

Key terms used across the CheckMAITE documentation. Each entry links to the page that covers it in depth.

## CheckMAITE

- **Harness**: CheckMAITE itself: a test and evaluation (T&E) harness that runs evaluation workflows over MAITE-compliant models, datasets, and metrics, and collects their results in a consistent form.
- **Capability**: One evaluation workflow the harness can run, such as model inference plus metrics on a dataset, or a dataset bias analysis. Each capability defines the configuration it accepts and the outputs it produces. See [Key concepts](../development/key_concepts.md#capability).
- **Run**: Everything associated with one execution of a capability: its configuration, its outputs, and an optional report. See [Key concepts](../development/key_concepts.md#run).
- **Analytics store**: Machine-readable results of capability runs, stored as one table per capability so tools and SQL queries can consume them across runs, datasets, and capabilities. This is separate from the human-readable reports a run produces. See [Save & query results](../get-started/saving_querying_results.md).
- **Record**: One flat, scalar-only row in an analytics store table, produced by a run's `extract()` method. See the [Analytics store guide](../development/analytics_store_guide.ipynb).
- **Job backend**: The execution target for asynchronous capability runs, selected with `configure_job_backend(...)`. CheckMAITE ships `ray` (registry-backed, reattachable jobs) and `ray-simple` (process-local Ray tasks). See [Job backend configuration](../development/job_submission/configure_job_backend.md).
- **Job**: A non-blocking handle to a submitted capability run, with lifecycle states and a reference-first `result()`. See [Job protocol and lifecycle](../development/job_submission/protocol.md).
- **Worker environment**: The software environment, such as a container image and its installed packages, that remote workers use to run capabilities. See [Worker environments](../development/job_submission/worker_environments.md).
- **Plugin**: An installable package that adds capabilities to CheckMAITE through entry points. See [Plugin system](../development/plugins.md).
- **Persona**: A representative user role, such as Data Scientist or ML Engineer, used to frame tutorials and guide design. See [Personas](personas.md).

## MAITE and data

- **MAITE**: The Modular AI Trustworthy Engineering protocols that CheckMAITE models, datasets, and metrics conform to. See the [MAITE documentation](https://mit-ll-ai-technology.github.io/maite/).
- **Wrapper**: A MAITE-compliant object around a model, dataset, or metric that CheckMAITE capabilities can consume. See [Conventions](conventions.md).
- **`index2label`**: The `dict[int, str]` map from class index to class name that models and datasets expose. Where a model's and a dataset's maps overlap, they should agree. Nothing checks this, and a mismatch gives silently wrong per-class results. See [Conventions](conventions.md#index2label-relationship-between-models-and-datasets).
- **Annotation format**: The on-disk dataset layout CheckMAITE can load, such as COCO, YOLO, or VisDrone. See [Supported dataset annotation formats](conventions.md#supported-dataset-annotation-formats).

## JATIC tools

- **DataEval**: JATIC's dataset analysis toolkit. In CheckMAITE it powers the bias, feasibility, cleaning (linting), and shift capabilities. See the [Dataeval bias tutorial](../tool-usage/dataeval_bias_tutorial.ipynb).
- **Bias**: Correlations or imbalances in a dataset that can lead a model to learn shortcuts. See the [Dataeval bias tutorial](../tool-usage/dataeval_bias_tutorial.ipynb).
- **Feasibility**: An estimate of the irreducible error of a dataset, which bounds how well any model can perform on it. See the [Dataeval feasibility tutorial](../tool-usage/dataeval_feasibility_tutorial.ipynb).
- **Cleaning (linting)**: Finding invalid, irrelevant, or low-quality data such as duplicates and statistical outliers. See the [Dataeval linting tutorial](../tool-usage/dataeval_linting_tutorial.ipynb).
- **Shift**: Statistical differences between operational data and training data. See the [Dataeval shift tutorial](../tool-usage/dataeval_shift_tutorial.ipynb).
- **NRTK (Natural Robustness Toolkit)**: Generates operationally realistic image perturbations to measure how model performance degrades. See the [NRTK tutorial](../tool-usage/nrtk_tutorial.ipynb).
- **Perturber**: An NRTK algorithm that modifies an image according to a fixed configuration, such as halving its brightness. See the [NRTK tutorial](../tool-usage/nrtk_tutorial.ipynb).
- **Perturbation factory**: An NRTK generator of perturbers that vary one or more parameters (theta keys) across a range. See the [NRTK tutorial](../tool-usage/nrtk_tutorial.ipynb).
- **XAITK (Explainable AI Toolkit)**: Generates visual saliency maps that show which image regions drive a model's prediction. See the [XAITK tutorial](../tool-usage/xaitk_tutorial.ipynb).
- **Saliency map**: A per-pixel importance map for one prediction. See the [XAITK tutorial](../tool-usage/xaitk_tutorial.ipynb).
