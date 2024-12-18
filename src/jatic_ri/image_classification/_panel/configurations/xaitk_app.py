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
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack

# local imports
from jatic_ri import PACKAGE_DIR
from jatic_ri._common._panel.xaitk_app_common import BaseXAITKApp

mpl.use("agg")

pn.extension("tabulator")
pn.extension("jsoneditor")

IMAGE_DIR = PACKAGE_DIR / "_sample_imgs" / "XAITK"
TEST_IMAGE = IMAGE_DIR / "example_car_img.jpg"


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
        input_tensor = torch.as_tensor(batch)
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
    title_font_size = param.Integer(default=24)
    status_text = param.String("Waiting for detection image input...")

    def __init__(self, **params: dict[str, object]) -> None:
        #   model initialization
        model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        self.jatic_classifier: ic.Model = HuggingFaceClassifier(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.classifier = JATICImageClassifier(
            classifier=self.jatic_classifier,
            ids=sorted(self.jatic_classifier.index2label),
        )
        # id2label mapping should be used from model protocol
        # metadata after relevant updates to protocols are added
        self.id2label = self.jatic_classifier.index2label

        self.md_text = """
                The sample saliency generation uses the [in21k](https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10)
                model and a sample image from the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset.
                To generate optimal saliency maps, kindly
                refer to the documentation:
                https://xaitk-saliency.readthedocs.io/en/latest/implementations.html#end-to-end-saliency-generation
                """

        super().__init__(**params)

        self.test_img = TEST_IMAGE
        self.select_widget = pn.widgets.Select(
            name="Choose Class",
            options={v: k for k, v in self.id2label.items()},
            stylesheets=[self.widget_stylesheet],
        )
        self.saliency_gen_button.on_click(self.saliency_gen_button_callback)

        self.sample_image = pn.pane.Matplotlib(self.create_sample_image(), tight=True)

    def _run_export(self) -> None:
        """This function runs when `export_button` is clicked"""
        widget_values = self.collect_widget_values()
        self.widget_values.append(widget_values)

        for idx, widget_value in enumerate(self.widget_values):
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
                },
            }

    def get_sal_plot_size(self) -> tuple(int, int):
        """Return size of saliency plot"""
        return (4, 3)

    def get_sal_plot_title(self) -> str:
        """Return str of saliency plot title"""
        return f"Saliency Map (class: {self.id2label[self.select_widget.value]})"

    def generate_saliency(self, img: np.ndarray) -> np.ndarray:
        """Method to generate saliency maps for a given saliency algorithm and classification model"""
        widget_value = self.collect_widget_values()
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
        img = np.asarray(Image.open(self.test_img))
        gray_img = np.asarray(Image.open(self.test_img).convert("L"))

        self.status_text = "Generating saliency maps..."
        sal_maps = self.generate_saliency(img)
        self.status_text = "Saliency maps created"

        sal_map = sal_maps[self.select_widget.value]
        self.augmented_plot.object = self.create_sal_plot(img=gray_img, sal_array=sal_map)
        self.augmented_plot.visible = True
        self.status_text = "Saliency generation test completed"

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
