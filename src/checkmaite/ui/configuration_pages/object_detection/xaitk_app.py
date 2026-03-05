"""XAITKApp for Object Detection.

This module contains the XAITKAppOD class, an implementation of BaseXAITKApp
for object detection. It allows configuration and creation of XAITKTestStage
instances for object detection tasks.

To save the app as HTML instead of serving it, run with the ``--ci`` flag.

"""

# Python generic imports

import os
import sys
import warnings
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# 3rd party and JATIC package imports
import maite.protocols.object_detection as od
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param
import torch
from bokeh.resources import INLINE
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image
from PIL.Image import Image as PilImg
from smqtk_core.configuration import to_config_dict
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from xaitk_jatic.interop.object_detection.model import JATICDetector
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack, RandomGridStack

from checkmaite.core._utils import set_device

# local imports
from checkmaite.ui._common.base_app import DEFAULT_STYLING, AppStyling
from checkmaite.ui._common.xaitk_app_common import BaseXAITKApp

mpl.use("agg")


IMAGE_DIR = IMAGE_DIR = Path(__file__).parents[3] / "assets" / "XAITK"
TEST_IMAGE = IMAGE_DIR / "XAITK_Visdrone_example_img.jpg"
STACK_OPTIONS = ["D-RISE", "RandomGrid"]


@dataclass
class DetectionTarget:
    """Dataclass conforming to MAITE's ObjectDetectionTarget.

    Attributes
    ----------
    boxes : torch.Tensor
        Bounding box coordinates.
    labels : torch.Tensor
        Class labels for each box.
    scores : torch.Tensor
        Confidence scores for each detection.

    """

    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


class HuggingFaceDetector:
    """MAITE wrapper for HuggingFace object detection models.

    Parameters
    ----------
    model_name : str
        Name of the HuggingFace model to load.
    threshold : float
        Detection threshold.
    device : str
        Device to run the model on (e.g., "cpu", "cuda").

    """

    def __init__(self, model_name: str, threshold: float, device: str) -> None:
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
        )

        # Upstream issue https://github.com/huggingface/transformers/issues/37615 with no response.
        # Assuming ignoring is not problematic for now.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="for .* copying from a non-meta parameter in the checkpoint to a meta parameter in the "
                "current model, which is a no-op\\.",
                category=UserWarning,
            )
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.threshold = threshold
        self.device = device

        self.model.eval()
        self.model.to(device)

    @property
    def index2label(self) -> dict[int, Hashable]:
        """Return mapping from class ID to label."""
        return self.model.config.id2label

    def __call__(self, batch: Sequence[od.InputType]) -> Sequence[od.TargetType]:
        """Perform inference on a batch of images.

        Parameters
        ----------
        batch : Sequence[od.InputType]
            A batch of input images.

        Returns
        -------
        Sequence[od.TargetType]
            A batch of detection targets.

        Raises
        ------
        ValueError
            If input tensor dimensions are invalid.

        """
        from torchvision.transforms.functional import get_image_size

        # tensor bridging
        input_tensor = torch.from_numpy(np.array(batch))
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

        predictions: Sequence[od.TargetType] = list()  # noqa: C408
        for result in hf_results:
            predictions.append(
                DetectionTarget(
                    result["boxes"].detach().cpu(),
                    result["labels"].detach().cpu(),
                    result["scores"].detach().cpu(),
                ),
            )

        return predictions


class XAITKAppOD(BaseXAITKApp):
    """App for building XAITKTestStages for object detection.

    This application allows users to configure and generate saliency maps
    for object detection models using XAITK.

    Parameters
    ----------
    styles : AppStyling, optional
        Styling configuration, by default DEFAULT_STYLING.
    **params : dict[str, object]
        Additional parameters for the `param.Parameterized` base class.

    Attributes
    ----------
    title : param.String
        The title of the application page.
    jatic_detector : od.Model
        The JATIC object detection model.
    detector : JATICDetector
        The XAITK-JATIC wrapper for the object detection model.
    id2label : dict
        Mapping from class ID to label name.
    pad_perc : float
        Padding percentage for cropping images around detections.
    stack_select : pn.widgets.Select
        Widget to select the saliency generation stack (e.g., D-RISE, RandomGrid).
    md_text : str
        Markdown text providing context about the sample saliency generation.
    test_img : Path
        Path to the test image.
    select_widget : pn.widgets.Select
        Widget to select a specific detection from the top-10 detections.
    saliency_gen_button : pn.widgets.Button
        Button to trigger saliency map generation.
    sample_image : pn.pane.Matplotlib
        Pane to display the sample image with optional bounding boxes.

    """

    title = param.String(default="Configure XAITK Saliency Generation Testing")  # pyright: ignore[reportAssignmentType]

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        #   model initialization
        model_name = "facebook/detr-resnet-50"
        self.jatic_detector: od.Model = HuggingFaceDetector(  # pyright: ignore[reportAttributeAccessIssue]
            model_name=model_name,
            threshold=0.5,
            device=set_device(None),  # pyright: ignore[reportArgumentType]
        )
        self.detector = JATICDetector(
            detector=self.jatic_detector,
            ids=sorted(self.jatic_detector.index2label.keys()),  # pyright: ignore[reportAttributeAccessIssue]
            img_batch_size=5,
        )
        # id2label mapping should be used from model protocol
        # metadata after relevant updates to protocols are added
        self.id2label = self.jatic_detector.index2label  # pyright: ignore[reportAttributeAccessIssue]
        self.pad_perc = 0.4

        self.stack_select = pn.widgets.Select(
            name="Choose generator",
            options=STACK_OPTIONS,
        )

        super().__init__(styles, **params)

        self.md_text = """
                The sample saliency generation uses the [DETR-Resnet50](https://huggingface.co/facebook/detr-resnet-50)
                model and a sample image from the [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) dataset.
                The default saliency configuration in this app takes about
                3 minutes to run on a server instance with a dual-core CPU
                and 8GB RAM. To generate optimal saliency maps, kindly
                refer to the [XAITK Documentation](https://xaitk-saliency.readthedocs.io/en/latest/implementations.html#end-to-end-saliency-generation).
                """
        self.test_img = TEST_IMAGE
        self.select_widget = pn.widgets.Select(
            name="Choose Detection (from top-10 detections)",
            options=self.get_top_10_detections(),
            stylesheets=[self.styles.widget_stylesheet],
        )
        self.saliency_gen_button.on_click(self.saliency_gen_button_callback)

        self.sample_image = pn.pane.Matplotlib(self.create_sample_image(bboxes=None), tight=True)

    def _run_export(self) -> None:
        """Collect configurations and prepare XAITKTestStage definitions.

        This method gathers widget values, configures the selected saliency
        generator (D-RISE or RandomGrid), and stores the test stage
        configurations in `self.output_test_stages`.
        """
        widget_values = self.collect_widget_values()
        self.widget_values.append(widget_values)
        generator_type = self.stack_select.value
        for idx, widget_value in enumerate(self.widget_values):
            fill = [95, 96, 93]
            if generator_type == "D-RISE":
                saliency_generator = DRISEStack(
                    n=widget_value["num_masks"],
                    s=widget_value["grid_size"][0],
                    p1=0.7,
                    threads=8,
                    seed=42,
                )
                saliency_generator.fill = fill
            elif generator_type == "RandomGrid":
                saliency_generator = RandomGridStack(
                    n=widget_value["num_masks"],
                    s=widget_value["grid_size"],
                    p1=0.7,
                    threads=8,
                    seed=42,
                )
                saliency_generator.fill = fill

            self.output_test_stages[f"{self.__class__.__name__}_{idx}"] = {  # pyright: ignore[reportIndexIssue]
                "TYPE": "XAITKTestStage",
                "CONFIG": {
                    "name": f"saliency_{self.__class__.__name__}_{idx}",
                    "saliency_generator": to_config_dict(saliency_generator),  # pyright: ignore[reportPossiblyUnboundVariable]
                    "img_batch_size": widget_value["img_batch_size"],
                },
            }

    def dets_to_mats_output(
        self, dets: Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert detection results to bounding box and score matrices.

        Parameters
        ----------
        dets : Iterable[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]]
            An iterable of detections, where each detection contains a bounding
            box and a dictionary of class scores.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - bboxes (np.ndarray): An array of bounding boxes [N, 4].
            - scores (np.ndarray): An array of class scores [N, num_classes].
        """
        labels = sorted(self.id2label.keys())

        bboxes = np.empty((0, 4))
        scores = np.empty((0, len(labels)))
        for det in dets:
            bbox = det[0]  # pyright: ignore[reportIndexIssue]

            bboxes = np.vstack(
                (
                    bboxes,
                    [
                        *bbox.min_vertex,
                        *bbox.max_vertex,
                    ],
                )
            )

            score_dict = det[1]  # pyright: ignore[reportIndexIssue]
            score_array = [score_dict[label] for label in labels]

            scores = np.vstack(
                (
                    scores,
                    score_array,
                ),
            )

        return bboxes, scores

    def get_top_10_detections(self) -> dict:
        """Retrieve the top-10 detections from the test image.

        Detections are sorted by their confidence scores in descending order.

        Returns
        -------
        dict
            A dictionary mapping a descriptive string of the detection
            (index, class name, score) to its original index in the
            full list of detections.
        """
        img = np.asarray(Image.open(self.test_img))
        dets = list(self.detector([img]))[0]
        confs = list()  # noqa: C408
        for det in dets:
            score_dict = det[1]
            cls_name = max(score_dict, key=score_dict.get)  # pyright: ignore[reportCallIssue, reportArgumentType]
            confs.append(score_dict[cls_name])

        indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        top_10_dets = [dets[i] for i in indices[0:10]]  # pyright: ignore[reportIndexIssue]
        dropdown_options = dict()  # noqa: C408
        for i, det in enumerate(top_10_dets):
            score_dict = det[1]
            cls_name = max(score_dict, key=score_dict.get)
            dropdown_options.update({f"{i} : {self.id2label[cls_name].title()}_{score_dict[cls_name]:.5f}": indices[i]})
        return dropdown_options

    def generate_saliency(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate saliency maps for a selected detection.

        This method uses the configured saliency generator (D-RISE or RandomGrid)
        and the object detection model to produce saliency maps for the
        detection chosen via `self.select_widget`.

        Parameters
        ----------
        img : np.ndarray
            The input image as a NumPy array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - sal_maps (np.ndarray): The generated saliency maps.
            - bboxes (np.ndarray): The bounding box of the selected detection.
        """
        widget_value = self.collect_widget_values()
        generator_type = self.stack_select.value

        fill = [95, 96, 93]
        if generator_type == "D-RISE":
            saliency_generator = DRISEStack(
                n=widget_value["num_masks"],
                s=widget_value["grid_size"][0],
                p1=0.7,
                threads=8,
                seed=42,
            )
            saliency_generator.fill = fill
        elif generator_type == "RandomGrid":
            saliency_generator = RandomGridStack(
                n=widget_value["num_masks"],
                s=widget_value["grid_size"],
                p1=0.7,
                threads=8,
                seed=42,
            )
            saliency_generator.fill = fill

        dets = list(self.detector([img]))[0]
        det = list(dets)[int(self.select_widget.value)]  # pyright: ignore[reportArgumentType]
        bboxes, scores = self.dets_to_mats_output([det])  # pyright: ignore[reportArgumentType]
        sal_maps = saliency_generator(img, bboxes, scores, self.detector)  # pyright: ignore[reportPossiblyUnboundVariable]

        return sal_maps, bboxes

    def saliency_gen_button_callback(self, _event: object) -> None:
        """Handle the saliency generation button click event.

        This callback generates saliency maps for the selected detection,
        updates the status bar, and displays the original detection crop
        and the corresponding saliency map.

        Parameters
        ----------
        _event : object
            The event object from the button click (unused).
        """
        img = np.asarray(Image.open(self.test_img))
        gray_img = np.asarray(Image.open(self.test_img).convert("L"))

        self.status_source.emit("Generating saliency maps...")
        sal_maps, bboxes = self.generate_saliency(img)
        self.status_source.emit("Saliency maps created")

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
        self.original_plot.object = self.create_det_plot(img=img_crop)
        self.original_plot.visible = True
        self.augmented_plot.object = self.create_sal_plot(img=grayscale_img_crop, sal_array=sal_crop)
        self.augmented_plot.visible = True
        self.status_source.emit("Saliency generation test completed")

    def create_det_plot(self, img: PilImg | None) -> Figure:
        """Create a Matplotlib figure displaying a cropped detection.

        Parameters
        ----------
        img : PilImg | None
            The cropped image of the detection. If None, a placeholder
            is used.

        Returns
        -------
        Figure
            A Matplotlib figure object.
        """
        img_array = np.asarray(img) if img is not None else np.zeros((5, 5))
        fig, ax = plt.subplots(figsize=self.get_sal_plot_size())
        ax.imshow(img_array)
        ax.set_title("Example Detection", fontdict={"fontsize": 6})
        ax.axis("off")
        plt.tight_layout()
        plt.close()
        return fig

    def create_sample_image(self, bboxes: np.ndarray | None, include_dets: bool = False) -> Figure:
        """Create a Matplotlib figure of the sample input image.

        Optionally, a bounding box for a selected detection can be overlaid.

        Parameters
        ----------
        bboxes : np.ndarray | None
            An array containing bounding box coordinates [x1, y1, x2, y2].
            If `include_dets` is True and `bboxes` is provided, the first
            bounding box is drawn.
        include_dets : bool, optional
            Whether to include the detection bounding box on the image,
            by default False.

        Returns
        -------
        Figure
            A Matplotlib figure object.
        """
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
        """Update the sample image plot to include detection bounding boxes.

        Parameters
        ----------
        bboxes : np.ndarray
            An array containing bounding box coordinates to display.
        """
        self.sample_image.object = self.create_sample_image(include_dets=True, bboxes=bboxes)

    @pn.depends("stack_select.value")
    def view_generator_params(self) -> pn.Column:
        """Return a Panel column containing generator selection and parameters.

        This view is dependent on the value of `stack_select`.

        Returns
        -------
        pn.Column
            A Panel column with the stack selector and saliency widget.
        """
        return pn.Column(self.stack_select, self.saliency_widget[0])

    def panel(self) -> pn.Column:
        """Construct the main Panel layout for the application.

        Returns
        -------
        pn.Column
            The main Panel layout.
        """
        return pn.Column(
            self.view_header,
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


if __name__ == "__main__":
    sd: XAITKAppOD = XAITKAppOD()
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
