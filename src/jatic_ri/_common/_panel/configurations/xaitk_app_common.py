"""
This module contains the XAITKApp class, which is an implementation of BaseApp.
It is able to configure and create multiple XAITKTestStage classes for consumption.

Run with `--ci` flag to save the app as html instead of serving it.
"""

# Python generic imports
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 3rd party and JATIC package imports
import panel as pn
import param
from matplotlib.figure import Figure
from PIL.Image import Image as PilImg

# local imports
from jatic_ri import PACKAGE_DIR
from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling, BaseApp

mpl.use("agg")

IMAGE_DIR = PACKAGE_DIR / "_sample_imgs" / "XAITK"
XAITK_LOGO = IMAGE_DIR / "XAITK_logo.png"


class BaseXAITKApp(BaseApp):
    """App for building XAITKTestStages"""

    md_text: str = param.String()

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)

        self.saliency_gen_button = pn.widgets.Button(
            name="Test Saliency Map Generation Settings",
            button_type="primary",
            stylesheets=[self.styles.button_stylesheet],
            align="end",
        )
        self.select_widget = pn.widgets.Select(
            name="Choose Class",
            options=[],
            stylesheets=[self.styles.widget_stylesheet],
        )

        self.original_plot = pn.pane.Matplotlib(
            self.create_sal_plot(img=None, sal_array=None), tight=True, visible=False
        )
        self.augmented_plot = pn.pane.Matplotlib(
            self.create_sal_plot(img=None, sal_array=None), tight=True, visible=False
        )

        self.left_column_width = 410
        self.saliency_widget = [
            pn.Card(
                self.add_saliency_gen_config_widget(),
                title="Saliency Generation Parameters",
                header_color=self.styles.color_gray_900,
                width=self.left_column_width,
            ),
        ]
        self.widget_values = []
        self.sample_image = self.original_plot

    def create_sal_plot(self, img: PilImg | None, sal_array: np.ndarray | None) -> Figure:
        """Return matplotlib figure of the original sample image with a saliency map"""
        img_array = np.asarray(img) if img is not None else np.zeros((5, 5))
        fig, ax = plt.subplots(figsize=self.get_sal_plot_size())
        if sal_array is not None:
            ax.imshow(img_array, alpha=0.7, cmap="gray")
            ax.imshow(sal_array, alpha=0.3, cmap="jet")
            ax.set_title(self.get_sal_plot_title(), fontdict={"fontsize": 6})
        ax.axis("off")
        plt.tight_layout()
        plt.close()
        return fig

    def get_sal_plot_size(self) -> tuple(int, int):
        """Return size of saliency plot"""
        return (2, 2)

    def get_sal_plot_title(self) -> str:
        """Return str of saliency plot title"""
        return "Saliency Map"

    def add_saliency_gen_config_widget(self) -> pn.Column:
        """Saliency Generation config widget"""
        return pn.Column(
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
            pn.pane.Markdown(
                f"""
                <style>
                * {{
                    color: {self.styles.color_gray_900};
                }}
                </style>
                {self.md_text}
                """,
            ),
        )

    def collect_widget_values(self) -> dict:
        """Collect all the values on the current widgets"""
        saliency_params = self.saliency_widget[0].objects[0].objects

        return {
            "num_masks": saliency_params[0].value,
            "grid_size": tuple(int(el) for el in saliency_params[1].value[1:-1].split(",")),
            "img_batch_size": saliency_params[2].value,
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
            stylesheets=[self.styles.text_color_styling],
        )

    def panel(self) -> pn.Column:
        """High level view of the full app"""
        return pn.Column(
            self.view_header,
            pn.Row(self.view_title, pn.layout.HSpacer(), self.view_logo),
            pn.Row(
                self.saliency_widget[0],
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
