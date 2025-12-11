"""
This module contains the XAITKApp class, which is an implementation of BaseApp.
It is able to configure and create multiple XAITKTestStage classes for consumption.
"""

# Python generic imports

import os
import sys
import warnings
from collections.abc import Hashable, Sequence
from pathlib import Path

# 3rd party and JATIC package imports
import maite.protocols.image_classification as ic
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import torch
from bokeh.resources import INLINE
from matplotlib.figure import Figure
from PIL import Image
from smqtk_core.configuration import to_config_dict
from xaitk_jatic.interop.image_classification.model import JATICImageClassifier
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise import MCRISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack

from jatic_ri.core._utils import set_device

# local imports
from jatic_ri.ui._common.base_app import DEFAULT_STYLING, AppStyling
from jatic_ri.ui._common.xaitk_app_common import BaseXAITKApp

mpl.use("agg")


IMAGE_DIR = Path(__file__).parents[3] / "assets" / "XAITK"
TEST_IMAGE = IMAGE_DIR / "example_car_img.jpg"
STACK_OPTIONS = ["RISE", "MC-RISE"]


class HuggingFaceClassifier:
    """
    MAITE wrapper for HuggingFaceClassifier.

    Parameters
    ----------
    model_name : str
        The name of the HuggingFace model to load.
    device : str
        The device to run the model on (e.g., "cpu", "cuda").

    """

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
        )

        # Upstream issue https://github.com/huggingface/transformers/issues/37615 with no response.
        # Assuming ignoring is not problematic for now.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="for conv1.weight. copying from a non-meta parameter in the checkpoint to a meta parameter "
                "in the current model, which is a no-op\\.",
                category=UserWarning,
            )
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.device = device

        self.model.eval()
        self.model.to(device)

    @property
    def index2label(self) -> dict[int, Hashable]:
        """
        Class id to label mapping.

        Returns
        -------
        dict[int, Hashable]
            A dictionary mapping class IDs to their corresponding labels.
        """
        return self.model.config.id2label

    def __call__(self, batch: Sequence[ic.InputType]) -> Sequence[ic.TargetType]:
        """
        Callable implementation for HuggingFaceClassifier.

        Parameters
        ----------
        batch : Sequence[ic.InputType]
            A batch of input images.

        Returns
        -------
        Sequence[ic.TargetType]
            A batch of target predictions.

        Raises
        ------
        ValueError
            If the input tensor does not have 4 dimensions.
        """
        # tensor bridging
        input_tensor = torch.as_tensor(np.array(batch))
        if input_tensor.ndim != 4:
            raise ValueError(f"Invalid input dimensions. Expected 4, got {input_tensor.ndim}")

        # preprocess
        hf_inputs = self.image_processor(input_tensor, return_tensors="pt")

        # put on device
        hf_inputs = hf_inputs.to(self.device)

        # get predictions
        with torch.no_grad():
            return self.model(**hf_inputs).logits.softmax(1).detach().cpu()


class XAITKAppIC(BaseXAITKApp):
    """
    App for building XAITKTestStages for image classification.

    Attributes
    ----------
    title : param.String
        The title of the application page.
    jatic_classifier : ic.Model
        The JATIC image classification model.
    classifier : JATICImageClassifier
        The XAITK-JATIC wrapped image classifier.
    id2label : dict[int, Hashable]
        Mapping from class ID to label.
    stack_select : pn.widgets.Select
        Widget to select the saliency generation stack (RISE or MC-RISE).
    test_img : Path
        Path to the test image.
    select_widget : pn.widgets.Select
        Widget to select the class for saliency map generation.
    sample_image : pn.pane.Matplotlib
        Panel pane to display the sample image.

    """

    title = param.String(default="Configure XAITK Saliency Generation Testing")

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        #   model initialization
        model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        self.jatic_classifier: ic.Model = HuggingFaceClassifier(model_name=model_name, device=set_device(None))
        self.classifier = JATICImageClassifier(
            classifier=self.jatic_classifier,
            ids=sorted(self.jatic_classifier.index2label),
        )
        # id2label mapping should be used from model protocol
        # metadata after relevant updates to protocols are added
        self.id2label = self.jatic_classifier.index2label

        self.stack_select = pn.widgets.Select(name="Choose generator", options=STACK_OPTIONS)

        super().__init__(styles, **params)

        self.stack_select.link(self.stack_select, callbacks={"value": self.stack_select_callback})

        self.test_img = TEST_IMAGE
        self.select_widget = pn.widgets.Select(
            name="Choose Class",
            options={v: k for k, v in self.id2label.items()},
            stylesheets=[self.styles.widget_stylesheet],
        )
        self.saliency_gen_button.on_click(self.saliency_gen_button_callback)
        self.stack_select.stylesheets = [self.styles.widget_stylesheet]

        self.sample_image = pn.pane.Matplotlib(self.create_sample_image(), tight=True)

    def stack_select_callback(self, target: object, _event: object) -> None:  # noqa ARG001
        """
        Callback for stack_select widget. Updates the saliency_widget.

        Parameters
        ----------
        target : object
            The object that triggered the event.
        _event : object
            The event object.
        """
        self.saliency_widget = [
            pn.Card(
                self.add_saliency_gen_config_widget(),
                title="Saliency Generation Parameters",
                header_color=self.styles.color_gray_900,
                width=self.left_column_width,
            )
        ]

    def add_saliency_gen_config_widget(self) -> pn.Column:
        """
        Create and return the saliency generation configuration widget.

        Returns
        -------
        pn.Column
            A Panel Column containing widgets for saliency generation parameters.
        """

        generator_type = self.stack_select.value
        basic_component = pn.Column(
            pn.widgets.IntInput(
                name="Number of Masks",
                value=50,
                start=50,
                end=1200,
                step=50,
                stylesheets=[self.styles.widget_stylesheet],
            ),
            pn.widgets.Select(
                name="Occlusion Grid Size",
                options=["(7,7)", "(5,5)", "(10,10)"],
                stylesheets=[self.styles.widget_stylesheet],
            ),
            pn.widgets.IntInput(
                name="Image Batch Size",
                value=1,
                start=1,
                step=1,
                stylesheets=[self.styles.widget_stylesheet],
            ),
        )
        if generator_type == "MC-RISE":
            basic_component.append(
                pn.widgets.LiteralInput(
                    name="Fill Colors",
                    value=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]],
                    type=list,
                    stylesheets=[self.styles.widget_stylesheet],
                )
            )
        basic_component.append(
            pn.pane.Markdown(
                f"""
                    <style>
                    * {{
                        color: {self.styles.color_gray_900};
                    }}
                    </style>
                    The sample saliency generation uses the [in21k](https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10)
                    model and a sample image from the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset.
                    To generate optimal saliency maps, kindly
                    refer to the documentation:
                    https://xaitk-saliency.readthedocs.io/en/latest/implementations.html#end-to-end-saliency-generation
                """
            )
        )

        return basic_component

    def _run_export(self) -> None:
        """
        Collect all configurations into a dictionary object.

        This dictionary is shared across app pages and defines the
        XAITKTestStage configurations.
        """
        widget_values = self.collect_widget_values()
        self.widget_values.append(widget_values)
        generator_type = self.stack_select.value

        for idx, widget_value in enumerate(self.widget_values):
            if generator_type == "MC-RISE":
                saliency_generator = MCRISEStack(
                    n=widget_value["num_masks"],
                    s=widget_value["grid_size"][0],
                    p1=0.7,
                    fill_colors=list(widget_value["fill_colors"]),
                    threads=8,
                    seed=42,
                )
            elif generator_type == "RISE":
                saliency_generator = RISEStack(
                    n=widget_value["num_masks"],
                    s=widget_value["grid_size"][0],
                    p1=0.7,
                    threads=8,
                    seed=42,
                )
                fill = [95, 96, 93]
                saliency_generator.fill = fill

            self.output_test_stages[f"{self.__class__.__name__}_{idx}"] = {
                "TYPE": "XAITKTestStage",
                "CONFIG": {
                    "name": f"saliency_{self.__class__.__name__}_{idx}",
                    "saliency_generator": to_config_dict(saliency_generator),
                    "img_batch_size": widget_value["img_batch_size"],
                },
            }

    def get_sal_plot_size(self) -> tuple[int, int]:
        """
        Return the size of the saliency plot.

        Returns
        -------
        tuple[int, int]
            A tuple representing the (width, height) of the saliency plot in inches.
        """
        return (4, 3)

    def get_sal_plot_title(self) -> str:
        """
        Return the title string for the saliency plot.

        Returns
        -------
        str
            The title for the saliency plot, including the selected class.
        """
        return f"Saliency Map (class: {self.id2label[self.select_widget.value]})"

    def collect_widget_values(self) -> dict:
        """
        Collect all the values from the current saliency configuration widgets.

        Returns
        -------
        dict
            A dictionary containing the current values of the saliency parameters.
        """
        saliency_params = self.saliency_widget[0].objects[0].objects
        generator_type = self.stack_select.value

        values = {
            "num_masks": saliency_params[0].value,
            "grid_size": tuple(int(el) for el in saliency_params[1].value[1:-1].split(",")),
            "img_batch_size": saliency_params[2].value,
        }

        if generator_type == "MC-RISE":
            values["fill_colors"] = saliency_params[3].value

        return values

    def generate_saliency(self, img: np.ndarray) -> np.ndarray:
        """
        Generate saliency maps for a given image, saliency algorithm, and classification model.

        Parameters
        ----------
        img : np.ndarray
            The input image as a NumPy array.

        Returns
        -------
        np.ndarray
            The generated saliency maps.
        """
        widget_value = self.collect_widget_values()
        generator_type = self.stack_select.value

        if generator_type == "MC-RISE":
            saliency_generator = MCRISEStack(
                n=widget_value["num_masks"],
                s=widget_value["grid_size"][0],
                p1=0.7,
                fill_colors=list(widget_value["fill_colors"]),
                threads=8,
                seed=42,
            )
        elif generator_type == "RISE":
            saliency_generator = RISEStack(
                n=widget_value["num_masks"],
                s=widget_value["grid_size"][0],
                p1=0.7,
                threads=8,
                seed=42,
            )
            fill = [95, 96, 93]
            saliency_generator.fill = fill
        return saliency_generator(np.asarray(img), self.classifier)

    def saliency_gen_button_callback(self, _event: object) -> None:
        """
        Callback for the saliency generation button.

        Generates saliency maps based on current configurations and displays the results.

        Parameters
        ----------
        _event : object
            The event object from the button click.
        """
        self.augmented_plot.visible = False
        img = np.asarray(Image.open(self.test_img))
        gray_img = np.asarray(Image.open(self.test_img).convert("L"))
        generator_type = self.stack_select.value

        self.status_source.emit("Generating saliency maps...")
        sal_maps = self.generate_saliency(img)
        self.status_source.emit("Saliency maps created")

        if generator_type == "MC-RISE":
            sal_map = sal_maps[:, self.select_widget.value, :, :]
            multiple_maps = pn.Column()
            widget_value = self.collect_widget_values()
            for color_map, fill_color in zip(sal_map, widget_value["fill_colors"], strict=False):
                multiple_maps.append(
                    pn.pane.Matplotlib(
                        self.create_sal_plot_mc_rise(img=gray_img, sal_array=color_map, title=fill_color), tight=True
                    )
                )
            self.augmented_plot = multiple_maps
        else:
            sal_map = sal_maps[self.select_widget.value]
            self.augmented_plot = pn.pane.Matplotlib(
                self.create_sal_plot(img=gray_img, sal_array=sal_map), tight=True, visible=False
            )
        self.augmented_plot.visible = True
        self.status_source.emit("Saliency generation test completed")

    def create_sal_plot_mc_rise(self, img: np.ndarray, sal_array: np.ndarray, title: str) -> Figure:
        """
        Create a Matplotlib figure for an MC-RISE saliency map.

        The figure shows the original sample image overlaid with a saliency map.

        Parameters
        ----------
        img : np.ndarray
            The original image (grayscale) as a NumPy array.
        sal_array : np.ndarray
            The saliency map array.
        title : str
            The title suffix for the plot (e.g., fill color).

        Returns
        -------
        Figure
            A Matplotlib Figure object.
        """
        img_array = np.asarray(img)
        fig, ax = plt.subplots(figsize=self.get_sal_plot_size())
        ax.imshow(img_array, alpha=0.7, cmap="gray")
        ax.imshow(sal_array, alpha=0.3, cmap="seismic")
        ax.set_title(f"{self.get_sal_plot_title()}  {title}", fontdict={"fontsize": 6})
        ax.axis("off")
        plt.tight_layout()
        plt.close()
        return fig

    def create_sample_image(self) -> Figure:
        """
        Create a Matplotlib figure of the sample input image.

        Returns
        -------
        Figure
            A Matplotlib Figure object displaying the sample image.
        """
        img = np.asarray(Image.open(self.test_img))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Sample Image")
        ax.axis("off")
        ax.imshow(img)
        plt.tight_layout()
        plt.close()
        return fig

    def update_plot(self) -> None:
        """Update the Matplotlib image/plot for the sample image."""
        self.sample_image.object = self.create_sample_image()

    @pn.depends("stack_select.value")
    def view_generator_params(self) -> pn.Column:
        """
        Return a Panel Column containing the generator selection and its parameters.

        This view depends on the value of `stack_select`.

        Returns
        -------
        pn.Column
            A Panel Column with the stack selector and saliency widget.
        """
        return pn.Column(self.stack_select, self.saliency_widget[0])

    def panel(self) -> pn.Column:
        """
        Construct and return the main Panel layout for the application.

        Returns
        -------
        pn.Column
            The main Panel Column representing the application's UI.
        """
        return pn.Column(
            pn.Row(self.view_title, pn.layout.HSpacer(), self.view_logo),
            pn.Row(
                self.view_generator_params,
                pn.Column(
                    pn.pane.Markdown(
                        f"""
                            <style>
                            * {{
                                color: {self.styles.color_gray_900};
                            }}
                            </style>
                            <h2> Choose Sample Detection
                        """
                    ),
                    self.select_widget,
                    self.sample_image,
                    pn.pane.Markdown(
                        f"""
                            <style>
                            * {{
                                color: {self.styles.color_gray_900};
                            }}
                            </style>
                            <h2> Saliency Generation Output
                        """
                    ),
                    self.view_plots,
                ),
            ),
            pn.layout.Divider(),
            pn.Row(pn.layout.HSpacer(), self.saliency_gen_button),
            self.view_status_bar,
            pn.Spacer(height=24),
            pn.Row(pn.HSpacer(), self.next_button),
            width=self.styles.app_width,
            styles={"background": self.styles.color_main_bg},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: XAITKAppIC = XAITKAppIC()
    app = sd.panel()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        app.save(os.path.join("artifacts", "xaitk_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr.write(msg)
        sys.exit(1)
    else:
        # special adaption to ensure the poetry blocks execution when the server is running
        server = pn.serve(app, address="localhost", port=5008, show=True, threaded=True)
        server.join()
