# Using JATIC tools

These tutorials follow the [T&E workflow](../reference/personas.md#te-workflow).
Each one names its persona and the workflow stage it covers.

## Dataset evaluation

<div class="grid cards" markdown >

- [__Dataeval: Bias__ :octicons-arrow-right-24:](dataeval_bias_tutorial.ipynb)

    The Dataeval bias detection tools identify biases or correlations present in
    a dataset that may lead to shortcut learning in models.

- [__Dataeval: Feasibility__ :octicons-arrow-right-24:](dataeval_feasibility_tutorial.ipynb)

    The Dataeval feasiblity tools measure the irreducible error of a dataset.

- [__Dataeval: Cleaning__ :octicons-arrow-right-24:](dataeval_linting_tutorial.ipynb)

    The Dataeval cleaning tools analyze datasets to identify and remove invalid,
    irrelevant, or low-quality data, such as duplicates and statistical
    outliers.

- [__Dataeval: Shift__ :octicons-arrow-right-24:](dataeval_shift_tutorial.ipynb)

    The Dataeval shift tool identifies statistical differences between
    operational data and training data

</div>

## Model evaluation

<div class="grid cards" markdown >

- [__NRTK__ :octicons-arrow-right-24:](nrtk_tutorial.ipynb)

    The Natural Robustness Toolkit (NRTK) is an open source toolkit for
    generating operationally realistic perturbations to evaluate the natural
    robustness of computer vision algorithms.

- [__XAITK__ :octicons-arrow-right-24:](xaitk_tutorial.ipynb)

    The Explainable AI Toolkit for saliency (XAITK - Saliency) is an open source
    framework for visual saliency algorithm interfaces and implementations.

</div>

## Results & analysis

<div class="grid cards" markdown >

- [__Analytics Store__ :octicons-arrow-right-24:](analytics_store_tutorial.ipynb)

    The analytics store persists capability results as queryable records,
    enabling SQL-based comparison across runs, datasets, and capabilities.

</div>

## Job submission

<div class="grid cards" markdown >

- [__Ray Simple Job Submission__ :octicons-arrow-right-24:](ray_simple_job_submission_tutorial.ipynb)

    Run a capability asynchronously with the lightweight process-local Ray task
    backend.

- [__Ray Job Submission__ :octicons-arrow-right-24:](ray_job_submission_tutorial.ipynb)

    Run a capability asynchronously with the registry-backed Ray backend for
    tracked and reattachable jobs.

</div>

## End-to-end workflows

<div class="grid cards" markdown >

- [__Object Detection Workflow via API__ :octicons-arrow-right-24:](../get-started/checkmaite_api_od.ipynb)

    Run every capability on one object detection model and dataset, then
    build a single report.

- [__Image Classification Workflow via API__ :octicons-arrow-right-24:](../get-started/checkmaite_api_ic.ipynb)

    The same end-to-end workflow for image classification.

</div>

For task-focused steps, see the [how-to guides](../how-to/index.md).
