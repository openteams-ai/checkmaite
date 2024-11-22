"""
This module contains the NRTKApp class, which is an implemntation of BaseApp.
It is able to configure and create NRTKTestStages for consumption

Run with `--ci` flag to save the app as html instead of serving it.
"""

# Python generic imports
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

# Panel app imports
import panel as pn
import param
from bokeh.resources import INLINE

# NRTK imports
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from PIL import Image

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# local imports
from jatic_ri import PACKAGE_DIR
from jatic_ri.object_detection._panel.configurations.base_app import BaseApp
from jatic_ri.object_detection.test_stages.impls.nrtk_test_stage import NRTKTestStage

pn.extension("tabulator")
pn.extension("jsoneditor")

IMAGE_DIR = PACKAGE_DIR / "_sample_imgs"
EXAMPLE_IMG = IMAGE_DIR / "nrtk_test_image.jpeg"
NRTK_LOGO = IMAGE_DIR / "NRTK_logo.png"


class NRTKApp(BaseApp):
    """App for building NRTKTestStages"""

    title = param.String(default="Configure Natural Robustness Testing")
    title_font_size = param.Integer(default=24)
    status_text = param.String("Waiting for input...")

    def __init__(self, **params: dict[str, object]) -> None:
        self.add_button = pn.widgets.Button(
            name="Add Test Stage", button_type="primary"
        )  # declare here since its used in pn.depends
        self.clear_button = pn.widgets.Button(
            name="Clear Test Stages", button_type="primary"
        )  # declare here since its used in pn.depends
        self.perturber_select = pn.widgets.Select(
            name="Perturber", options={pert_impl.__name__: pert_impl for pert_impl in PerturbImage.get_impls()}
        )

        super().__init__(**params)
        self.add_button.stylesheets = [self.button_stylesheet]
        self.clear_button.stylesheets = [self.button_stylesheet]
        self.perturber_select.stylesheets = [self.widget_stylesheet]

        self.add_button.on_click(self.add_test_stage_callback)
        self.clear_button.on_click(self.clear_test_stage_callback)

        self.test_perturber_button = pn.widgets.Button(
            name="Test Perturber Settings", button_type="primary", stylesheets=[self.button_stylesheet]
        )
        self.test_perturber_button.on_click(self.test_perturber_button_callback)

        self.original_plot = pn.pane.Matplotlib(self.create_original_plot(), tight=True)
        self.augmented_plot = pn.pane.Matplotlib(self.create_original_plot(), tight=True, visible=False)

        self.left_column_width = 410

        self._pybsm_parameter_init()

        def single_perturber_callback(target: object, _event: object) -> None:  # noqa ARG001
            self.all_widgets = [
                pn.Card(
                    self.add_perturber_config_widget(),
                    title="Factory Configuration",
                    header_color=self.color_light_gray,
                )
            ]

        self.all_widgets = [
            pn.Card(
                self.add_perturber_config_widget(),
                title="Factory Configuration",
                header_color=self.color_light_gray,
                width=self.left_column_width,
            )
        ]

        self.test_stages = []
        self.finished_factory_display = []

        self.perturber_select.link(self.perturber_select, callbacks={"value": single_perturber_callback})

    def _run_export(self) -> None:
        """This function runs when `export_button` is clicked"""
        if len(self.test_stages) == 0:
            self.status_text = "No configurations found. Press Add Test Stage to add a configuration."
            return
        for idx, stage in enumerate(self.test_stages):
            stage_name = stage["name"]
            self.output_test_stages[f"{self.__class__.__name__}_{idx}"] = {
                "TYPE": NRTKTestStage,
                "CONFIG": {
                    "name": f"natural_robustness_{stage_name}",
                    "perturber_factory": stage["factory"],
                },
            }

    def add_perturber_config_widget(self) -> pn.Column:
        """Add perturber factory config widget"""
        pert_impl = self.perturber_select.value

        factory_options = {}
        self.name_input = pn.widgets.TextInput(name="Stage Name", placeholder="Enter a name here...")
        self.name_input.stylesheets = [self.widget_stylesheet]

        if pert_impl.__name__ != "PybsmPerturber":
            for factory_impl in PerturbImageFactory.get_impls():
                # Skip Pybsm factory and private factory
                if "Pybsm" in factory_impl.__name__:
                    continue
                factory_options[factory_impl.__name__] = factory_impl
        else:
            for factory_impl in PerturbImageFactory.get_impls():
                # Skip Pybsm factory and private factory
                if factory_impl.__name__ == "CustomPybsmPerturbImageFactory":
                    factory_options[factory_impl.__name__] = factory_impl

        self.factory_selector = pn.widgets.Select(name="Factory Type", options=factory_options)
        self.factory_selector.stylesheets = [self.widget_stylesheet]

        if pert_impl.__name__ == "PybsmPerturber":
            self.theta_keys_input = pn.widgets.LiteralInput(name="Theta Keys", type=list)
            self.theta_keys_input.stylesheets = [self.widget_stylesheet]
            self.thetas_input = pn.widgets.LiteralInput(name="Thetas", type=list)
            self.thetas_input.stylesheets = [self.widget_stylesheet]
            return pn.Column(
                self.name_input,
                self._setup_scenario_parameters(),
                self._setup_sensor_parameters(),
                self.theta_keys_input,
                self.thetas_input,
            )

        return pn.Column(self.factory_selector, self.name_input, self.factory_config)

    @pn.depends("factory_selector.value")
    def factory_config(self) -> dict:
        """Get factory config"""
        pert_impl = self.perturber_select.value
        factory_impl = self.factory_selector.value

        if factory_impl.__name__ == "StepPerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=list(pert_impl.get_default_config().keys()))
            self.theta_key.stylesheets = [self.widget_stylesheet]
            self.start = pn.widgets.FloatInput(name="Start", value=0.0)
            self.start.stylesheets = [self.widget_stylesheet]
            self.stop = pn.widgets.FloatInput(name="Stop", value=1.0)
            self.stop.stylesheets = [self.widget_stylesheet]
            self.step = pn.widgets.FloatInput(name="Step", value=1.0)
            self.step.stylesheets = [self.widget_stylesheet]
            self.to_int = pn.widgets.Checkbox(name="Return integers", value=True)
            self.to_int.stylesheets = [self.widget_stylesheet]

            return pn.Column(self.theta_key, self.start, self.stop, self.step, self.to_int)
        if factory_impl.__name__ == "LinSpacePerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=list(pert_impl.get_default_config().keys()))
            self.theta_key.stylesheets = [self.widget_stylesheet]
            self.start = pn.widgets.FloatInput(name="Start", value=0.0)
            self.start.stylesheets = [self.widget_stylesheet]
            self.stop = pn.widgets.FloatInput(name="Stop", value=1.0)
            self.stop.stylesheets = [self.widget_stylesheet]
            self.step = pn.widgets.IntInput(name="Step", value=1)
            self.step.stylesheets = [self.widget_stylesheet]

            return pn.Column(self.theta_key, self.start, self.stop, self.step)
        if factory_impl.__name__ == "OneStepPerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=list(pert_impl.get_default_config().keys()))
            self.theta_key.stylesheets = [self.widget_stylesheet]
            self.theta_value = pn.widgets.FloatInput(name="Theta Value", value=0.0)
            self.theta_value.stylesheets = [self.widget_stylesheet]

            return pn.Column(self.theta_key, self.theta_value)
        bad_factory_text = pn.widget.StaticText(value=f"{factory_impl.__name__} is not supported")
        bad_factory_text.stylesheets = [self.widget_stylesheet]
        return pn.Column(bad_factory_text)

    def _pybsm_parameter_init(self) -> None:
        self.altitude_provider = pn.widgets.FloatInput(name="Altitude (m)")
        self.altitude_provider.description = (
            "Sensor height above ground level in meters. The database includes the following "
            "altitude options: 2m 32.55m 75m 150m 225m 500m, 1000m to 12000m in 1000m steps, "
            "14000m to 20000m in 2000m steps, and 24500m"
        )
        self.altitude_provider.value = 75
        self.altitude_provider.stylesheets = [self.widget_stylesheet]

        self.ground_range_provider = pn.widgets.FloatInput(name="Ground Range (m)")
        self.ground_range_provider.description = (
            "Distance on the ground between the target and sensor in meters. The following "
            "ground ranges are included in the database at each altitude until the ground "
            "range exceeds the distance to the spherical earth horizon: 0m 100m 500m, 1000m to "
            "20000m in 1000m steps, 22000m to 80000m in 2000m steps, and 85000m to "
            "300000m in 5000m steps."
        )
        self.ground_range_provider.value = 0
        self.ground_range_provider.stylesheets = [self.widget_stylesheet]

        self.scenario_name_provider = pn.widgets.TextInput(name="Scenario Name")
        self.scenario_name_provider.value = ""

        self.ihaze_provider = pn.widgets.Select(name="IHAZE", options=[1, 2])
        self.ihaze_provider.description = "MODTRAN code for visibility"
        self.ihaze_provider.value = 2

        self.aircraft_speed_provider = pn.widgets.FloatInput(name="Aircraft Speed (m/s)")
        self.aircraft_speed_provider.description = "Ground speed of the aircraft"
        self.aircraft_speed_provider.value = 1000

        self.target_reflectance_provider = pn.widgets.FloatInput(name="Target Reflectance")
        self.target_reflectance_provider.description = "Object reflectance"
        self.target_reflectance_provider.value = 0.15

        self.target_temperature_provider = pn.widgets.FloatInput(name="Target Temperature (K)")
        self.target_temperature_provider.description = "Object temperature (Kelvin)"
        self.target_temperature_provider.value = 295

        self.background_reflectance_provider = pn.widgets.FloatInput(name="Background Reflectance")
        self.background_reflectance_provider.description = "Background reflectance"
        self.background_reflectance_provider.value = 0.07

        self.background_temperature_provider = pn.widgets.FloatInput(name="Background Temperature (K)")
        self.background_temperature_provider.description = "Used to calculate the turbulence profile"
        self.background_temperature_provider.value = 293

        self.ha_windspeed_provider = pn.widgets.FloatInput(name="High Altitude Windspeed (m/s)")
        self.ha_windspeed_provider.description = "Background reflectance"
        self.ha_windspeed_provider.value = 21

        self.cn2at1m_provider = pn.widgets.FloatInput(name="Refractive Index Structure Parameter")
        self.cn2at1m_provider.description = (
            'The refractive index structure parameter "near the ground" (e.g. '
            "at h = 1m). Used to calculate the turbulence profile"
        )
        self.cn2at1m_provider.value = 0

        self.d_provider = pn.widgets.FloatInput(name="Effective Aperture Diameter (m)")
        self.d_provider.value = 0.005
        self.d_provider.stylesheets = [self.widget_stylesheet]

        self.f_provider = pn.widgets.FloatInput(name="Focal Length (m)")
        self.f_provider.value = 0.014
        self.f_provider.stylesheets = [self.widget_stylesheet]

        self.sensor_name_provider = pn.widgets.TextInput(name="Sensor Name")
        self.sensor_name_provider.value = ""

        self.px_provider = pn.widgets.FloatInput(name="Detector Center-to-Center Spacing (Pitch) (m)")
        self.px_provider.value = 0.0000074

        self.opt_trans_wavelengths_provider = pn.widgets.LiteralInput(
            name="Spectral Bandpass of the Camera (m)", type=list
        )
        self.opt_trans_wavelengths_provider.description = "At minimum, a start and end wavelength should be specified"
        self.opt_trans_wavelengths_provider.value = [3.8e-07, 7e-07]

        self.optics_transmission_provider = pn.widgets.LiteralInput(
            name="Full System In-Band Optical Transmission", type=list
        )
        self.optics_transmission_provider.description = "Loss due to any telescope obscuration should not be included"
        self.optics_transmission_provider.value = []

        self.eta_provider = pn.widgets.FloatInput(name="Relative Linear Obscuration")
        self.eta_provider.value = 0.4

        self.int_time_provider = pn.widgets.FloatInput(name="Integration Time (s)")
        self.int_time_provider.description = "Maximum integration time"
        self.int_time_provider.value = 0.03

        self.dark_current_provider = pn.widgets.FloatInput(name="Detector Dark Current (e-/s)")
        self.dark_current_provider.value = 0

        self.read_noise_provider = pn.widgets.FloatInput(name="RMS Read Noise (RMS e-)")
        self.read_noise_provider.value = 25.0

        self.max_N_provider = pn.widgets.FloatInput(name="Maximum ADC Level (e-)")
        self.max_N_provider.value = 100000000

        self.bit_depth_provider = pn.widgets.FloatInput(name="Bit Depth (bits)")
        self.bit_depth_provider.description = "Resolution of the detector ADC"
        self.bit_depth_provider.value = 16

        self.max_well_fill_provider = pn.widgets.FloatInput(name="Max Well Fill")
        self.max_well_fill_provider.description = "Desired well fill. i.e. maximum well size x desired fill fraction"
        self.max_well_fill_provider.value = 1.0

        self.s_x_provider = pn.widgets.FloatInput(name="RMS Jitter Amplitude, X Direction (rad)")
        self.s_x_provider.value = 0

        self.s_y_provider = pn.widgets.FloatInput(name="RMS Jitter Amplitude, Y Direction (rad)")
        self.s_y_provider.value = 0

        self.da_x_provider = pn.widgets.FloatInput(name="Line of Sight Angular Drift Rate, X Direction (rad/s)")
        self.da_x_provider.description = "Drift rate during one integration time"
        self.da_x_provider.value = 0.0

        self.da_y_provider = pn.widgets.FloatInput(name="Line of Sight Angular Drift Rate, Y Direction (rad/s)")
        self.da_y_provider.description = "Drift rate during one integration time"
        self.da_y_provider.value = 0.0

        self.qe_provider = pn.widgets.LiteralInput(
            name="Quantum Efficiency as a function of Wavelength (e-/photon)", type=list
        )
        self.qe_provider.value = [0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0]

        self.qewavelengths_provider = pn.widgets.LiteralInput(
            name="Wavelengths Corresponding to the Quantum Efficiency Array (microns)", type=list
        )
        self.qewavelengths_provider.value = [3e-07, 4e-07, 5e-07, 6e-07, 7e-07, 8e-07, 9e-07, 1e-06, 1.1e-06]

    def _setup_scenario_parameters(self) -> pn.Column:
        additional_params = pn.Card(
            self.scenario_name_provider,
            self.ihaze_provider,
            self.aircraft_speed_provider,
            self.target_reflectance_provider,
            self.background_reflectance_provider,
            self.target_temperature_provider,
            self.background_temperature_provider,
            self.ha_windspeed_provider,
            self.cn2at1m_provider,
            title="Scenario Parameters",
            collapsed=True,
            header_color=self.color_light_gray,
        )

        for widget in additional_params.objects:
            widget.stylesheets = [self.widget_stylesheet]

        return pn.Column(self.altitude_provider, self.ground_range_provider, additional_params)

    def _setup_sensor_parameters(self) -> pn.Column:
        additional_params = pn.Card(
            self.sensor_name_provider,
            self.px_provider,
            self.opt_trans_wavelengths_provider,
            self.optics_transmission_provider,
            self.eta_provider,
            self.int_time_provider,
            self.dark_current_provider,
            self.read_noise_provider,
            self.max_N_provider,
            self.bit_depth_provider,
            self.max_well_fill_provider,
            self.s_x_provider,
            self.s_y_provider,
            self.da_x_provider,
            self.da_y_provider,
            self.qe_provider,
            self.qewavelengths_provider,
            title="Sensor Parameters",
            collapsed=True,
            header_color=self.color_light_gray,
        )

        additional_params.header_color = self.color_light_gray

        for widget in additional_params.objects:
            widget.stylesheets = [self.widget_stylesheet]

        return pn.Column(self.d_provider, self.f_provider, additional_params)

    def add_test_stage_callback(self, _event: object) -> None:
        """Creates a new purturber widget when add_button is clicked"""
        factory_config = self.build_factory_json()
        if len(factory_config) == 0:
            return
        self.test_stages.append({"name": self.name_input.value, "factory": factory_config})
        theta_keys = (
            factory_config[factory_config["type"]]["theta_key"]
            if "theta_key" in factory_config[factory_config["type"]]
            else factory_config[factory_config["type"]]["theta_keys"]
        )
        self.finished_factory_display.append(
            pn.pane.Markdown(
                f"""
            <style>
            * {{
              color: {self.color_light_gray};
            }}
            </style>
            * Stage Name: {self.name_input.value}
            * Perturber Name: {self.perturber_select.value.__name__}
            * Factory Name: {self.factory_selector.value.__name__}
            * Theta Key(s): {theta_keys}
            """
            )
        )

    def clear_test_stage_callback(self, _event: object) -> None:
        """Clears all the stored widgets when the clear_button is clicked"""
        self.test_stages = []
        self.finished_factory_display = []

    def create_original_plot(self) -> plt.Figure:
        """Create plot of base image"""
        # Configure path to datasets
        img = np.asarray(Image.open(EXAMPLE_IMG))
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_title("Original Image")
        ax.axis("off")
        ax.imshow(img)
        plt.close()
        return fig

    def test_perturber_button_callback(self, _event: object) -> None:  # noqa PT019
        """Run final factory and display results"""
        self.status_text = "Perturbing..."
        factory_config = self.build_factory_json()
        if len(factory_config) == 0:
            return
        factory = from_config_dict(factory_config, PerturbImageFactory.get_impls())
        perturber = factory[len(factory) - 1]
        img = np.asarray(Image.open(EXAMPLE_IMG))
        perturbed_img = perturber(image=img, additional_params={"img_gsd": 3.19 / 160})
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis("off")
        ax.imshow(perturbed_img)
        ax.set_title("Augmented Image")
        plt.close()
        self.augmented_plot.object = fig
        self.augmented_plot.visible = True
        self.status_text = "Finished Perturbing"

    def _parse_pybsm_factory_config(self) -> (dict, dict):
        scenario_config = {
            "name": self.scenario_name_provider.value,
            "ihaze": self.ihaze_provider.value,
            "altitude": self.altitude_provider.value,
            "ground_range": self.ground_range_provider.value,
            "aircraft_speed": self.aircraft_speed_provider.value,
            "target_reflectance": self.target_reflectance_provider.value,
            "target_temperature": self.target_temperature_provider.value,
            "background_reflectance": self.background_reflectance_provider.value,
            "background_temperature": self.background_temperature_provider.value,
            "ha_wind_speed": self.ha_windspeed_provider.value,
            "cn2_at_1m": self.cn2at1m_provider.value,
        }

        opt_trans_wavelengths = self.opt_trans_wavelengths_provider.value
        optics_transmission = self.optics_transmission_provider.value
        optics_transmission = optics_transmission if optics_transmission else None
        qe = self.qe_provider.value
        qe = qe if qe else None
        qewavelengths = self.qewavelengths_provider.value
        qewavelengths = qewavelengths if qewavelengths else None

        sensor_config = {
            "name": self.sensor_name_provider.value,
            "D": self.d_provider.value,
            "f": self.f_provider.value,
            "p_x": self.px_provider.value,
            "opt_trans_wavelengths": opt_trans_wavelengths,
            "optics_transmission": optics_transmission,
            "eta": self.eta_provider.value,
            "int_time": self.int_time_provider.value,
            "dark_current": self.dark_current_provider.value,
            "read_noise": self.read_noise_provider.value,
            "max_n": self.max_N_provider.value,
            "bit_depth": self.bit_depth_provider.value,
            "max_well_fill": self.max_well_fill_provider.value,
            "s_x": self.s_x_provider.value,
            "s_y": self.s_y_provider.value,
            "da_x": self.da_x_provider.value,
            "da_y": self.da_y_provider.value,
            "qe": qe,
            "qe_wavelengths": qewavelengths,
        }

        return scenario_config, sensor_config

    def build_factory_json(self) -> dict:
        """Collect all the values on the current widgets"""
        pert_impl = self.perturber_select.value
        factory_impl = self.factory_selector.value
        if factory_impl.__name__ == "CustomPybsmPerturbImageFactory":
            scenario_config, sensor_config = self._parse_pybsm_factory_config()
            output_config = {}

            output_config["sensor"] = {
                "type": f"{PybsmSensor.__module__}.{PybsmSensor.__name__}",
                f"{PybsmSensor.__module__}.{PybsmSensor.__name__}": sensor_config,
            }
            output_config["scenario"] = {
                "type": f"{PybsmScenario.__module__}.{PybsmScenario.__name__}",
                f"{PybsmScenario.__module__}.{PybsmScenario.__name__}": scenario_config,
            }

            thetas = self.thetas_input.value
            theta_keys = self.theta_keys_input.value
            if len(thetas) != len(theta_keys):
                self.status_text = "Thetas and theta keys are different lengths"
                return {}
            output_config["thetas"] = thetas
            output_config["theta_keys"] = theta_keys
        elif factory_impl.__name__ == "StepPerturbImageFactory":
            output_config = {
                "perturber": f"{pert_impl.__module__}.{pert_impl.__name__}",
                "theta_key": self.theta_key.value,
                "start": self.start.value,
                "stop": self.stop.value,
                "step": self.step.value,
                "to_int": self.to_int.value,
            }
        elif factory_impl.__name__ == "LinSpacePerturbImageFactory":
            output_config = {
                "perturber": f"{pert_impl.__module__}.{pert_impl.__name__}",
                "theta_key": self.theta_key.value,
                "start": self.start.value,
                "stop": self.stop.value,
                "step": self.step.value,
            }
        elif factory_impl.__name__ == "OneStepPerturbImageFactory":
            output_config = {
                "perturber": f"{pert_impl.__module__}.{pert_impl.__name__}",
                "theta_key": self.theta_key.value,
                "theta_value": self.theta_value.value,
            }
        else:
            self.status_text = f"{factory_impl.__name__} is not supported"
            return {}

        return {
            "type": f"{factory_impl.__module__}.{factory_impl.__name__}",
            f"{factory_impl.__module__}.{factory_impl.__name__}": output_config,
        }

    @pn.depends("add_button.clicks", "clear_button.clicks", "perturber_select.value")
    def view_sweep_params(self) -> pn.Row:
        """When the add new perturber button is clicked or the "clear pertubers"
        button is clicked, this will trigger and update the view of the widgets"""
        # using pn.Card here to match the look of the collapsible config section
        return pn.Row(
            pn.Spacer(width=5),  # added spacer for visual separation
            pn.Column(
                pn.Column(*self.all_widgets),
                pn.layout.Divider(),
                pn.Row(
                    self.add_button,
                    self.clear_button,
                ),
                pn.layout.Divider(),
                pn.Column(*self.finished_factory_display),
            ),
        )

    def view_plots(self) -> pn.Column:
        """View of the plots"""
        return pn.Column(self.original_plot, self.augmented_plot)

    def view_logo(self) -> pn.pane.Image:
        """View NRTK logo"""
        return pn.pane.Image(
            str(NRTK_LOGO),
            width=140,
            styles={"display": "block", "float": "right", "background-color": "rgba(255, 255, 255, 1.0)"},
            stylesheets=[self.text_color_styling],
        )

    def panel(self) -> pn.Column:
        """High level view of the full app"""
        left_column = pn.Column(
            self.perturber_select,
            pn.Spacer(height=10),  # added spacer for visual separation
            self.view_sweep_params,
            width=self.left_column_width,
        )
        right_column = pn.Column(self.view_plots)
        return pn.Column(
            pn.Row(self.view_title, pn.layout.HSpacer(), self.view_logo),
            pn.Row(
                left_column,
                right_column,
            ),
            pn.Row(pn.layout.HSpacer(), self.export_button, self.test_perturber_button),
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_dark_blue},
        )


if __name__ == "__main__":  # pragma: no cover
    sd: NRTKApp = NRTKApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        os.makedirs("artifacts", exist_ok=True)
        sd.panel().save(os.path.join("artifacts", "nrtk_app.html"), resources=INLINE)
    elif len(sys.argv) > 1:
        msg = f"Got unexpected flag: {sys.argv[1]}"
        sys.stderr(msg)
        sys.exit(1)
    else:
        pn.serve(sd.panel(), host="127.0.0.1", port=5008)
