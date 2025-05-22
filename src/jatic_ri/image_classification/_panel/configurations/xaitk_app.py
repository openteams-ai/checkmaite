"""
This module contains the XAITKApp class, which is an implementation of BaseApp.
It is able to configure and create multiple XAITKTestStage classes for consumption.
"""

# Python generic imports
from __future__ import annotations

import os
import sys
from collections.abc import Hashable

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

# local imports
from jatic_ri import PACKAGE_DIR
from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling
from jatic_ri._common._panel.configurations.xaitk_app_common import BaseXAITKApp
from jatic_ri._common.models import set_device

mpl.use("agg")


IMAGE_DIR = PACKAGE_DIR / "_sample_imgs" / "XAITK"
TEST_IMAGE = IMAGE_DIR / "example_car_img.jpg"
STACK_OPTIONS = ["RISE", "MC-RISE"]


class HuggingFaceClassifier:
    """MAITE wrapper for HuggingFaceClassifier"""

    def __init__(self, model_name: str, device: str) -> None:
        from transformers import (
            AutoImageProcessor,  # type: ignore
            AutoModelForImageClassification,  # type: ignore
        )

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.device = device

        self.model.eval()
        self.model.to(device)

    @property
    def index2label(self) -> dict[int, Hashable]:
        """Class id to label mapping"""
        return self.model.config.id2label

    def __call__(self, batch: ic.InputBatchType) -> ic.TargetBatchType:
        """Callable implementation for HuggingFaceClassifier"""
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


class XAITKApp(BaseXAITKApp):
    """App for building XAITKTestStages for image classification"""

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
        self.saliency_widget = [
            pn.Card(
                self.add_saliency_gen_config_widget(),
                title="Saliency Generation Parameters",
                header_color=self.styles.color_gray_900,
                width=self.left_column_width,
            )
        ]

    def add_saliency_gen_config_widget(self) -> pn.Column:
        """Saliency Generation config widget"""

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
        """This function collects all configurations in a dictionary object
        that is shared across app pages."""
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

    def get_sal_plot_size(self) -> tuple(int, int):
        """Return size of saliency plot"""
        return (4, 3)

    def get_sal_plot_title(self) -> str:
        """Return str of saliency plot title"""
        return f"Saliency Map (class: {self.id2label[self.select_widget.value]})"

    def collect_widget_values(self) -> dict:
        """Collect all the values on the current widgets"""
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
        """Method to generate saliency maps for a given saliency algorithm and classification model"""
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
        Callback for saliency_gen_button.
        Generate saliency and display results
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
        """Return matplotlib figure of the original sample image with a saliency map"""
        img_array = np.asarray(img)
        fig, ax = plt.subplots(figsize=self.get_sal_plot_size())
        ax.imshow(img_array, alpha=0.7, cmap="gray")
        ax.imshow(sal_array, alpha=0.3, cmap="jet")
        ax.set_title(f"{self.get_sal_plot_title()}  {title}", fontdict={"fontsize": 6})
        ax.axis("off")
        plt.tight_layout()
        plt.close()
        return fig

    def create_sample_image(self) -> Figure:
        """View Sample Input Image"""
        img = np.asarray(Image.open(self.test_img))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Sample Image")
        ax.axis("off")
        ax.imshow(img)
        plt.tight_layout()
        plt.close()
        return fig

    def update_plot(self) -> None:
        """Update Matplotlib Image/Plot"""
        self.sample_image.object = self.create_sample_image()

    @pn.depends("stack_select.value")
    def view_generator_params(self) -> pn.Column:
        """Generator params helper"""
        return pn.Column(self.stack_select, self.saliency_widget[0])

    def panel(self) -> pn.Column:
        """High level view of the full app"""
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
            width=self.styles.app_width,
            styles={"background": self.styles.color_main_bg},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: XAITKApp = XAITKApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "xaitk_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
