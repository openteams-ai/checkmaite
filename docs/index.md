# Checkmaite Documentation

Checkmaite is an API and a UI application which makes testing and evaluation of models and datasets straightforward and reproducible. It can be used to run a wide variety of **Model Evaluation** and **Dataset Analysis** investigations for both **Object Detection** and **Image Classification** computer vision problems. 

It was built as an integration point for all [CDAO JATIC](https://cdao.pages.jatic.net/public/) tools and has since expanded focus. Our goal is to make T&E easier for analysts!

## Who is checkmaite for?

* **Low code users** - Our UI users are typically interested in testing and evaluation, likely have a good understanding of the domain, but may or may not be Data Scientists. They interact with checkmaite through the UI applications we provide.
* **No prior knowledge of specific JATIC tools** - Our users may or may not be familiar with the individual [JATIC products](https://cdao.pages.jatic.net/public/products/). They are familiar with their datasets and models and simply want to perform testing without needing to learn a new tool (or suite of tools).
* **Have their own models and datasets** - Our users are expected to come with arbitrary models and datasets. Among other out-of-the-box models, we enable loading default weights from torchvision while also providing a framework for users to bring their own. 

## Quick start

<div class="grid cards" markdown>

-   :material-monitor-arrow-down-variant:{ .lg .middle } [__Setup__ :octicons-arrow-right-24:](get-started/install_setup.md)

    ---
    Get your machine set up to run checkmaite.


-   :material-rocket-launch-outline:{ .lg .middle } [__Interactive Access to `checkmaite`__ :octicons-arrow-right-24:](get-started/checkmaite_interactive.md)

    ---
    Run the checkmaite dashboards locally or deploy on a server.

-  :material-connection:{ .lg .middle } [__Object Detection Workflow via API__ :octicons-arrow-right-24:](get-started/checkmaite_api_od.html)

    ---
    API access for Object Detection workflows.

-  :material-connection:{ .lg .middle } [__Image Classification Workflow via API__ :octicons-arrow-right-24:](get-started/checkmaite_api_ic.html)

    ---
    API access for Image Classification workflows.

-   :material-bookshelf:{ .lg .middle } [__Explore JATIC tools__ :octicons-arrow-right-24:](tool-usage/index.md)

    ---
    Learn to use the JATIC tools via checkmaite


</div>

## About the JATIC program and CDAO

The Joint AI Test Infrastructure Capability (JATIC) program develops software products for AI Test & Evaluation (T&E) and AI Assurance. The program is managed by the Assessment & Assurance Division of the [DoD Chief Digital and Artificial Intelligence Office (CDAO)](https://www.ai.mil/). It is funded from FY23-FY29.


!!! type "Program Mission"

    Develop software to accelerate and enable AI model test and evaluation for testers across the Department of Defense (DoD) enterprise, including DoD programs, research laboratories, industry partners, and academia in order to provide insight on the performance, effectiveness, robustness, and safety of the DoD's AI-enabled systems.

Learn more in the [CDAO JATIC program documentation](https://cdao.pages.jatic.net/public/).
