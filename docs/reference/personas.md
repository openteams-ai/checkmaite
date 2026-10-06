# User Personas

CheckMAITE's Python API supports several personas with varying skill levels. The
personas below represent the core
users we support.

## Data Scientist

### Data Scientist: who they are

* Researchers / analytics / data scientists working with dataset and models
* Most comfortable using Jupyter Notebooks, but can read/write some Python code
  as needed
* Focused on analysis and experimentation, not infrastructure

### Data Scientist: key workflows

* Complete T&E workflows using JATIC tools via CheckMAITE on models and datasets
* Explore and analyze data using notebooks
* Use cloud resources when local compute isn't enough
* Needs to compute a standard set of analyses in a structured manner

### Data Scientist: pain points

* Struggles to understand all the details of the wide variety of JATIC tools,
  wants a low barrier to entry with sane defaults
* No reproducibility or traceability on completed analyses
* Limited by computational power of local machine
* Collaboration with team members is difficult across local environments and
  operating systems
* Unclear what models and datasets are available to them

### Data Scientist: what they need

* Low barrier to entry for new tooling
* Jupyter Notebook workflows with clear, reusable Python examples
* Reproducibility of workflows
* Ability to search previously executed workflows
* High level guidance on JATIC tool usage through reasonable defaults and documentation
* Ability to run workflows on external resources (e.g. cloud)
* Ability to discovery models and datasets available

## ML Engineer

### ML Engineer: who they are

* ML Engineers / Software Engineers
* Most comfortable working with code via IDEs and CLIs
* Conducts T&E analyses
* Assists infrastructure engineers with day-to-day platform maintenance
  regarding user-facing tooling

### ML Engineer: key workflows

* Analyses datasets and models using Python
* Conducts their own T&E analyses using JATIC tools via CheckMAITE
* Enables Data Scientists by writing code to enable new workflows and simplify
  tool interactions
* Maintains model registry
* Maintains model served on platform

### ML Engineer: pain points

* Occassionally needs high compute resources
* Collaboration with team members is difficult across local environments and
  operating systems

### ML Engineer: what they need

* Ability to run workflows on external resources (e.g. cloud)
* Ability to discovery models and datasets available
* Deep understanding of JATIC tools in order to build interfaces for data scientists

## T&E workflow

A typical CheckMAITE project curates a new dataset, then uses it to pick the
most suitable of the available models.
Each tutorial covers one stage of that workflow.

### 1. Bring your model and dataset

**Led by:** ML Engineer

Wrap the model and dataset so CheckMAITE can run them. Every later stage depends
on this step.
Tutorial: [ONNX object detection wrapper](../tool-usage/onnx_object_detection_wrapper.ipynb).

### 2. Curate the dataset

**Led by:** Data Scientist

Check that the dataset is fit to evaluate models against before trusting any
model result on it. Remove low-quality
data, look for biases and correlations, compare it with operational data, and
check that the task is achievable.
Tutorials: [Cleaning](../tool-usage/dataeval_linting_tutorial.ipynb), [Bias](../tool-usage/dataeval_bias_tutorial.ipynb),
[Shift](../tool-usage/dataeval_shift_tutorial.ipynb), [Feasibility](../tool-usage/dataeval_feasibility_tutorial.ipynb).

### 3. Evaluate the model

**Led by:** Data Scientist

Measure each candidate model's baseline performance, how it holds up under
realistic perturbations, and what drives its predictions, so the team can choose
which model to fine-tune.
Tutorials: [MAITE Evaluation](../tool-usage/maite_evaluation_tutorial.ipynb),
[NRTK](../tool-usage/nrtk_tutorial.ipynb),
[XAITK](../tool-usage/xaitk_tutorial.ipynb).

### 4. Record and compare results

**Led by:** Data Scientist

Store results from every stage so they can be compared across runs, datasets,
and models without re-running anything.
Tutorial: [Analytics Store](../tool-usage/analytics_store_tutorial.ipynb).

### 5. Run at scale

**Led by:** ML Engineer, with Data Scientists using it when local compute isn't
enough

Move stages 2–4 off the local machine when they need more compute or have to be
shared, and serve models remotely.
Tutorials: [Ray Simple Job Submission](../tool-usage/ray_simple_job_submission_tutorial.ipynb),
[Ray Job Submission](../tool-usage/ray_job_submission_tutorial.ipynb),
[Remote Inference with Ray Serve](../get-started/checkmaite_ray_serve.ipynb).
