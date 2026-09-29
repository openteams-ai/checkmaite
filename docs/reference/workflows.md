# Workflows

CheckMAITE supports T&E analyses utilizing JATIC tooling through its Python API. This workflow can be adapted to different user groups and use cases.

Several aspects of the workflow are under active development.

!!! info "Feature Status"
    - [x] = Implemented
    - [ ] = Not yet implemented

## Python API Workflow

### Key Points

* High code access to JATIC tools provided through a unified python interface
* Flexible interface to enable construction of unique workflows and configurations to suite a wide variety of usecases

### Walkthrough

1. Create an environment (or use an existing one on the platform)
2. Open a python kernel (notebook, ipython, or write the following in a script)
3. Create model object(s)
4. Create dataset object(s)
5. Create metric object
6. Create cabability object
7. Create configuration object (if needed)
8. Execute the analysis
9. View the results

### Target Audience

* ML Engineers / Software Engineers 
* Users wanting deeper access to JATIC tools with a unified interface
* Users wanting more control over execution and configuration

### Local vs Deployed

This workflow has a few implementation changes depending on if the workflow is run locally or in a deployed, multi-user environment. 

#### Local

- [x] User creates a python script 
- [x] Models are stored locally (i.e. not being served anywhere)
- [x] Datasets are stored locally
- [x] Execution happens on the same machine that is running the script

#### Deployed platform

- [x] Can be run via python script, REST API :material-clock-outline:{ title="Planned for future release" }, or Jupyter Notebook
- [ ] Models are served on the platform (User queries the model service to discover available models)
- [ ] Datasets are served on the platform (User queries the dataset service to discover available datasets)
- [ ] Execution happens on a separate server from the one its being launched from
