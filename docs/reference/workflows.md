# Workflows

The RI supports T&E analyses utilizing JATIC tooling through several different workflows. Each workflow may be appropriate for a different user group or a different usecase. 

Several aspects of the workflow are under active development.

!!! info "Feature Status"
    - [x] = Implemented
    - [ ] = Not yet implemented

## UI Workflow

### Key Points
* User-friendly, low code introduction to JATIC tools
* May be run within a Jupyter Notebook, deployed locally, or deployed as a web app on an external server


### Walkthrough

Below is a typical walkthough of a user interacting with the UI to perform T&E analytics using JATIC tools.

1. Launch the dashboard (or navigates to a deployment of the dashboard)
2. Select the tools to be applied.
3. Configure tools through the interface (only applicable to some tools)
4. Define model(s) to be analyzed
5. Define dataset(s) to be analyzed
6. Click "Run Analysis" to begin the execution of all the analyses
7. Once execution is complete, review the output results

### Target Audience

* Data Scientists and junior ML Engineers
* Program leads
* Demo drivers running a demo of JATIC tools
* Users wanting to see the JATIC tools in action but not interested in a deep dive into each tool just yet
* Users wanting to conduct T&E analysis using JATIC tools in a simplified interface

### Local vs Deployed

This workflow has a few implementation changes depending on if the workflow is run locally or in a deployed, multi-user environment. 

#### Local

- [x] User is expected to launch the dashboard. Options are to deploy the web app via the Panel CLI or run inside of a Jupyter Notebook. 
- [x] Models are stored locally (i.e. not being served anywhere)
- [x] Datasets are stored locally
- [x] Execution happens on the same machine that is running the UI

#### Deployed platform

- [x] Launching the dashboard may be completed by the user or by a system admin (in which case user may just navigate to the web app). Running in a Jupyter Notebook is also an option. 
- [ ] Models are served on the platform (UI queries the model service for available options and presents those to the user)
- [ ] Datasets are served on the platform (UI queries the datasets that are available on the platform and presents those to the user)
- [ ] Execution happens on a separate server from the UI server

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
- [x] Execution happens on the same machine that is running the UI

#### Deployed platform

- [x] Can be run via python script, REST API :material-clock-outline:{ title="Planned for future release" }, or Jupyter Notebook
- [ ] Models are served on the platform (User queries the model service to discover available models)
- [ ] Datasets are served on the platform (User queries the dataset service to discover available datasets)
- [ ] Execution happens on a separate server from the one its being launched from
