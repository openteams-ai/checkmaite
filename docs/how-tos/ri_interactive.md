# Running the RI application interactively

We've streamlined the process of running the Reference Implementation (RI) application
interactively. This allows you to explore and utilize the full capabilities of the JATIC tools and workflows through a user-friendly web interface (UI).

To run the RI application, we recommend setting up a virtual environment and installing
the necessary dependencies, for detailed instructions, please refer to the [Setup
Guide](https://jatic.pages.jatic.net/reference-implementation/reference-implementation/community/setup.html).


Once you have the environment set up, you can start the RI application by running the
following command in your terminal:

```bash
poetry run panel serve src/jatic_ri/ui/app.py --show
```

</details>
If running a conda environment, you can replace the need for `poetry run` with `conda run -n <env_name> --live-stream`,
or if the environment is activated, you can simply use:

```bash
python panel serve src/jatic_ri/ui/app.py --show
```
</details>

This command will launch the RI application in your default web browser under
`http://localhost:5006/app`. If this port is in use, you can also specify the port, for example, `--port 9000`. You can
then interact with the application, explore the models, and run workflows directly from
the web interface.

The first step is to select which Workflow you would like to work with:

- *Model Evaluation (ME)* to analyze model performance, robustness, and explainability using the JATIC tools MAITE, NRTK, and XAITK respectively.
- *Dataset Analysis (DA)* to understand and improve dataset quality by analyzing data biases, feasability, shift, and cleaning statistics using the JATIC tool Dataeval. 

All of these tools are available for both Image Classification (IC) and Object Detection (OD) tasks.

![Preview of the RI Landing Page](../assets/ri_landing_page.png)

You can then select the desired workflow, and either load a pre-configured workflow parametrization JSON file or you can create a new configuration for this workflow.

Each configuration page will guide you through the necessary steps to set up the
workflow including any parameters
required for the workflow execution. Below is an example of the Model Evaluation (ME)
landing page, where you select which tools you would like to use:
![Model Evaluation Configuration](../assets/ri_me_config.png)

At end end of the configuration pipeline you
will be presented
with a summary of the configuration as a JSON block, and the options to select the model
and datasets to use for the workflow.

You can then run the workflow by clicking the "Run analysis"
button. The application will then execute the workflow and generate a report with the results.
![Model Evaluation Summary](../assets/ri_me_summary.png)

The Dataset Analysis workflow works in the same manner with different JATIC tools available. 

For detailed information on how each tool is configured and operates, refer to the individual pages in the RI documentation or the original tool’s documentation.
