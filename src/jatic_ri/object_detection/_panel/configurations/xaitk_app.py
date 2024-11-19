"""
This module contains the XAITKApp class, which is an implementation of BaseApp.
It is able to configure and create multiple XAITKTestStage classes for consumption.
"""

# Python generic imports
from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass

# 3rd party and JATIC package imports
import maite.protocols.object_detection as od
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from PIL.Image import Image as PilImg
from smqtk_core.configuration import to_config_dict
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from torchvision.transforms.functional import get_image_size  # type: ignore
from transformers import (
    AutoImageProcessor,  # type: ignore
    AutoModelForObjectDetection,  # type: ignore
)
from xaitk_jatic.interop.object_detection.model import JATICDetector
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import RandomGridStack

# local imports
from jatic_ri import PACKAGE_DIR
from jatic_ri.object_detection._panel.configurations.base_app import BaseApp

mpl.use("agg")

pn.extension("tabulator")
pn.extension("jsoneditor")

IMAGE_DIR = PACKAGE_DIR / "_sample_imgs" / "XAITK"
TEST_IMAGE = IMAGE_DIR / "XAITK_Visdrone_example_img.jpg"
XAITK_LOGO = IMAGE_DIR / "XAITK_logo.png"


@dataclass
class DetectionTarget:
    """Dataclass that conforms with MAITE's ObjectDetectionTarget"""

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class HuggingFaceDetector:
    """MAITE wrapper for HuggingFaceDetector"""

    def __init__(self, model_name: str, threshold: float, device: str) -> None:
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold
        self.device = device

        self.model.eval()
        self.model.to(device)

    def id2label(self) -> dict[int, Hashable]:
        """Class id to label mapping"""
        return self.model.config.id2label

    def __call__(self, batch: od.InputBatchType) -> od.TargetBatchType:
        """Callable implementation for HuggingFaceDetector"""
        # tensor bridging
        input_tensor = torch.as_tensor(batch)
        if input_tensor.ndim != 4:
            raise ValueError(f"Invalid input dimensions. Expected 4, got {input_tensor.ndim}")

        # save original image sizes for resizing boxes
        target_sizes = [get_image_size(img)[::-1] for img in input_tensor]

        # preprocess
        hf_inputs = self.image_processor(input_tensor, return_tensors="pt")

        # put on device
        hf_inputs = hf_inputs.to(self.device)

        # get predictions
        with torch.no_grad():
            hf_predictions = self.model(**hf_inputs)
        hf_results = self.image_processor.post_process_object_detection(
            hf_predictions,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )

        predictions: od.TargetBatchType = list()  # noqa: C408
        for result in hf_results:
            predictions.append(
                DetectionTarget(
                    result["boxes"].detach().cpu(),
                    result["labels"].detach().cpu(),
                    result["scores"].detach().cpu(),
                ),
            )

        return predictions


class XaitkApp(BaseApp):
    """App for building XAITKTestStages"""

    title = param.String(default="Configure XAITK Saliency Generation Testing")
    title_font_size = param.Integer(default=24)
    status_text = param.String("Waiting for detection image input...")

    default_xaitk_params = param.Dict({})

    def __init__(self, **params: dict[str, object]) -> None:
        #   model initialization
        model_name = "facebook/detr-resnet-50"
        self.jatic_detector: od.Model = HuggingFaceDetector(
            model_name=model_name,
            threshold=0.5,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.detector = JATICDetector(
            detector=self.jatic_detector,
            id_to_name=self.jatic_detector.id2label(),
            img_batch_size=5,
        )
        # id2label mapping should be used from model protocol
        # metadata after relevant updates to protocols are added
        self.id2label = self.jatic_detector.id2label()
        self.pad_perc = 0.4

        self.config_upload = pn.widgets.FileInput(accept=".json")  # declare here since its used in pn.depends

        super().__init__(**params)
        self.config_upload.stylesheets = [self.widget_stylesheet]

        self.test_img = TEST_IMAGE
        self.select_detection_widget = pn.widgets.Select(
            name="Choose Detection (from top-10 detections)",
            options=self.get_top_10_detections(),
            stylesheets=[self.widget_stylesheet],
        )
        self.saliency_gen_button = pn.widgets.Button(
            name="Test Saliency Map Generation Settings",
            button_type="primary",
            stylesheets=[self.button_stylesheet],
            align="end",
        )
        self.saliency_gen_button.on_click(self.saliency_gen_button_callback)

        self.original_plot = pn.pane.Matplotlib(
            self.create_sal_plot(img=None, sal_array=None), tight=True, visible=False
        )
        self.augmented_plot = pn.pane.Matplotlib(
            self.create_sal_plot(img=None, sal_array=None), tight=True, visible=False
        )

        self.left_column_width = 410
        self.right_column_width = 610
        self.page_width = self.left_column_width + self.right_column_width
        self.saliency_widget = [
            pn.Card(
                self.add_saliency_gen_config_widget(),
                title="Saliency Generation Parameters",
                header_color=self.color_light_gray,
                width=self.left_column_width,
            ),
        ]

        self.sample_image = pn.pane.Matplotlib(self.create_sample_image(bboxes=None), tight=True)
        self.widget_values = []

    def _run_export(self) -> None:
        """This function runs when `export_button` is clicked"""
        widget_values = self.collect_widget_values()
        self.widget_values.append(widget_values)

        for idx, widget_value in enumerate(self.widget_values):
            saliency_generator = RandomGridStack(
                n=widget_value["num_masks"],
                s=widget_value["grid_size"],
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
                    "id2label": self.id2label,
                },
            }

    def add_saliency_gen_config_widget(self) -> pn.Column:
        """Saliency Generation config widget"""
        return pn.Column(
            pn.widgets.IntInput(
                name="Number of Masks",
                value=50,
                start=50,
                end=1200,
                step=50,
                stylesheets=[self.widget_stylesheet],
            ),
            pn.widgets.Select(
                name="Occlusion Grid Size",
                options=["(7,7)", "(5,5)", "(10,10)"],
                stylesheets=[self.widget_stylesheet],
            ),
            pn.pane.Markdown(
                f"""
                <style>
                * {{
                    color: {self.color_light_gray};
                }}
                </style>
                The sample saliency generation uses the [DETR-Resnet50](https://huggingface.co/facebook/detr-resnet-50)
                model and a sample image from the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset.
                The default saliency configuration in this app takes about
                3 minutes to run on a server instance with a dual-core CPU
                and 8GB RAM. To generate optimal saliency maps, kindly
                refer to the documentation:
                https://xaitk-saliency.readthedocs.io/en/latest/implementations.html#end-to-end-saliency-generation
                """,
            ),
        )

    def create_sal_plot(self, img: PilImg | None, sal_array: np.ndarray | None) -> Figure:
        """Return matplotlib figure of the original sample image"""
        img_array = np.asarray(img) if img is not None else np.zeros((5, 5))
        fig, ax = plt.subplots(figsize=(2, 2))
        if sal_array is not None:
            ax.imshow(img_array, alpha=0.7, cmap="gray")
            ax.imshow(sal_array, alpha=0.3, cmap="jet")
            ax.set_title("Saliency Map", fontdict={"fontsize": 6})
        else:
            ax.imshow(img_array)
            ax.set_title("Example Detection", fontdict={"fontsize": 6})
        ax.axis("off")
        plt.tight_layout()
        plt.close()
        return fig

    def dets_to_mats_output(
        self, dets: Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Output detection and score matrices"""
        labels = [self.id2label[idx] for idx in sorted(self.id2label.keys())]

        bboxes = np.empty((0, 4))
        scores = np.empty((0, len(labels)))
        for det in dets:
            bbox = det[0]

            bboxes = np.vstack(
                (
                    bboxes,
                    [
                        *bbox.min_vertex,
                        *bbox.max_vertex,
                    ],
                )
            )

            score_dict = det[1]
            score_array = [score_dict[label] for label in labels]

            scores = np.vstack(
                (
                    scores,
                    score_array,
                ),
            )

        return bboxes, scores

    def get_top_10_detections(self) -> dict:
        """Retrieve the top-10 detections based on conf scores"""
        img = np.asarray(Image.open(self.test_img))
        dets = list(self.detector([img]))[0]
        confs = list()  # noqa: C408
        for det in dets:
            score_dict = det[1]
            cls_name = max(score_dict, key=score_dict.get)
            confs.append(score_dict[cls_name])

        indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        top_10_dets = [dets[i] for i in indices[0:10]]
        dropdown_options = dict()  # noqa: C408
        for i, det in enumerate(top_10_dets):
            score_dict = det[1]
            cls_name = max(score_dict, key=score_dict.get)
            dropdown_options.update({f"{i} : {cls_name.title()}_{score_dict[cls_name]:.5f}": indices[i]})
        return dropdown_options

    def generate_saliency(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Method to generate saliency maps for a given saliency algorithm and detection model"""
        widget_value = self.collect_widget_values()
        saliency_generator = RandomGridStack(
            n=widget_value["num_masks"],
            s=widget_value["grid_size"],
            p1=0.7,
            threads=8,
            seed=42,
        )
        fill = [95, 96, 93]
        saliency_generator.fill = fill

        dets = list(self.detector([img]))[0]
        det = list(dets)[int(self.select_detection_widget.value)]
        bboxes, scores = self.dets_to_mats_output([det])
        sal_maps = saliency_generator(img, bboxes, scores, self.detector)

        return sal_maps, bboxes

    def saliency_gen_button_callback(self, _event: object) -> None:
        """
        Callback for saliency_gen_button.
        Generate saliency and display results
        """
        img = np.asarray(Image.open(self.test_img))
        gray_img = np.asarray(Image.open(self.test_img).convert("L"))

        sal_maps, bboxes = self.generate_saliency(img)
        self.status_text = "Saliency maps created"

        sal_map = sal_maps[0]
        x1, y1, x2, y2 = bboxes[0]
        pad_x = int(round(self.pad_perc * (x2 - x1)))
        pad_y = int(round(self.pad_perc * (y2 - y1)))
        x1 = max(int(x1 - pad_x), 0)
        y1 = max(int(y1 - pad_y), 0)
        x2 = int(x2 + pad_x)
        y2 = int(y2 + pad_y)
        self.update_plot(bboxes=bboxes)
        img_crop = Image.fromarray(img[y1 : (y2 + 1), x1 : (x2 + 1)])
        grayscale_img_crop = Image.fromarray(gray_img[y1 : (y2 + 1), x1 : (x2 + 1)])
        sal_crop = sal_map[y1 : (y2 + 1), x1 : (x2 + 1)]
        self.original_plot.object = self.create_sal_plot(img=img_crop, sal_array=None)
        self.original_plot.visible = True
        self.augmented_plot.object = self.create_sal_plot(img=grayscale_img_crop, sal_array=sal_crop)
        self.augmented_plot.visible = True
        self.status_text = "Saliency generation test completed"

    def collect_widget_values(self) -> dict:
        """Collect all the values on the current widgets"""
        saliency_params = self.saliency_widget[0].objects[0].objects

        return {
            "num_masks": saliency_params[0].value,
            "grid_size": tuple(int(el) for el in saliency_params[1].value[1:-1].split(",")),
        }

    def view_plots(self) -> pn.Row:
        """View of the plots"""
        return pn.Row(self.original_plot, self.augmented_plot)

    def view_logo(self) -> pn.pane.Image:
        """View XAITK logo"""
        return pn.pane.Image(
            str(XAITK_LOGO),
            width=140,
            styles={"display": "block", "float": "right", "background-color": "rgba(255, 255, 255, 1.0)"},
            stylesheets=[self.text_color_styling],
        )

    def create_sample_image(self, bboxes: np.ndarray | None, include_dets: bool = False) -> Figure:
        """View Sample Input Image"""
        img = np.asarray(Image.open(self.test_img))
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("Sample Image")
        ax.axis("off")
        if include_dets and bboxes is not None:
            x1, y1, x2, y2 = bboxes[0]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"))
        ax.imshow(img)
        plt.tight_layout()
        plt.close()
        return fig

    def update_plot(self, bboxes: np.ndarray) -> None:
        """Update Matplotlib Image/Plot"""
        self.sample_image.object = self.create_sample_image(include_dets=True, bboxes=bboxes)

    def panel(self) -> pn.Column:
        """High level view of the full app"""
        return pn.Column(
            pn.Row(self.view_title, pn.layout.HSpacer(), self.view_logo),
            pn.Row(
                self.saliency_widget[0],
                pn.Column(
                    pn.pane.Markdown(
                        f"""
                            <style>
                            * {{
                                color: {self.color_light_gray};
                            }}
                            </style>
                            <h2> Choose Sample Detection
                        """
                    ),
                    self.select_detection_widget,
                    self.sample_image,
                    pn.pane.Markdown(
                        f"""
                            <style>
                            * {{
                                color: {self.color_light_gray};
                            }}
                            </style>
                            <h2> Saliency Generation Output
                        """
                    ),
                    self.view_plots,
                ),
            ),
            pn.layout.Divider(),
            pn.Row(pn.layout.HSpacer(), self.saliency_gen_button, self.export_button),
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_dark_blue},
        )


sd = XaitkApp()
sd.panel().servable()
