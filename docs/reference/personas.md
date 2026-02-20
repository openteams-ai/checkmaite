# User Personas

The RI has multiple interfaces which can be utilized by several different personas of varying skills. The personas 
below represent the core users we support. 

## Data Scientist

### Who They Are

* Researchers / analytics / data scientists working with dataset and models
* Most comfortable using Jupyter Notebooks or other UI frameworks, but can read/write some Python code as needed
* Focused on analysis and experimentation, not infrastructure

### Key Workflows

* Complete T&E workflows using JATIC tools via the RI on models and datasets
* Explore and analyze data using notebooks
* Use cloud resources when local compute isn't enough
* Needs to compute a standard set of analyses in a structured manner

### Pain Points

* Struggles to understand all the details of the wide variety of JATIC tools, wants a low barrier to entry with sane defaults
* No reproducibility or traceability on completed analyses
* Limited by computational power of local machine
* Collaboration with team members is difficult across local environments and operating systems
* Unclear what models and datasets are available to them

### What they need
* Low barrier to entry for new tooling
* Jupyter Notebook and/or UI interfaces to execute workflows
* Reproducibility of workflows
* Ability to search previously executed workflows
* High level guidance on JATIC tool usage through reasonable defaults and documentation
* Ability to run workflows on external resources (e.g. cloud)
* Ability to discovery models and datasets available

## ML Engineer

### Who They Are

* ML Engineers / Software Engineers
* Most comfortable working with code via IDEs and CLIs
* Conducts T&E analyses
* Assists infrastructure engineers with day-to-day platform maintenance regarding user-facing tooling


### Key Workflows

* Analyses datasets and models using Python
* Conducts their own T&E analyses using JATIC tools via the RI 
* Enables Data Scientists by writing code to enable new workflows and simplify tool interactions
* Maintains model registry
* Maintains model served on platform

### Pain Points

* Occassionally needs high compute resources
* Collaboration with team members is difficult across local environments and operating systems

### What they need
* Ability to run workflows on external resources (e.g. cloud)
* Ability to discovery models and datasets available
* Deep understanding of JATIC tools in order to build interfaces for data scientists
