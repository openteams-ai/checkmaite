"""
This module contains the NRTKBaseApp class, which is an implemntation of BaseApp.
It holds all the common code that can be used to create more specific NRTK Panel
apps.
It is able to configure and create NRTKTestStages for consumption

Run with `--ci` flag to save the app as html instead of serving it.
"""

# Python generic imports

# Type imports
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

# Panel app imports
import panel as pn
import param
from numpy.typing import NDArray
from PIL import Image

# SMQTK imports
from smqtk_core.configuration import from_config_dict

# local imports
from jatic_ri.ui._panel._common.base_app import DEFAULT_STYLING, AppStyling, BaseApp

IMAGE_DIR = JATIC_LOGO_PATH = Path(__file__).parents[3] / "assets"
EXAMPLE_IMG = IMAGE_DIR / "nrtk_test_image.jpeg"
NRTK_LOGO = IMAGE_DIR / "NRTK_logo.png"


class NRTKBaseApp(BaseApp):
    """App for building NRTKTestStages.

    Parameters
    ----------
    title : param.String
        The title of the application.
    next_parameter : param.Selector
        Selector for the next stage in the workflow.

    Attributes
    ----------
    add_button : pn.widgets.Button
        Button to add a test stage.
    clear_button : pn.widgets.Button
        Button to clear all test stages.
    perturber_select : pn.widgets.Select
        Widget to select the perturber implementation.
    test_perturber_button : pn.widgets.Button
        Button to test the current perturber settings.
    original_plot : pn.pane.Matplotlib
        Pane to display the original image.
    augmented_plot : pn.pane.Matplotlib
        Pane to display the augmented image.
    left_column_width : int
        Width of the left column in the layout.
    all_widgets : list
        List of all widgets for perturber configuration.
    test_stages : list
        List to store configured test stages.
    finished_factory_display : list
        List to display information about finished factories.
    name_input : pn.widgets.TextInput
        Input for the stage name.
    factory_selector : pn.widgets.Select
        Selector for the factory type.
    theta_keys_input : pn.widgets.LiteralInput
        Input for theta keys (for PybsmPerturber).
    thetas_input : pn.widgets.LiteralInput
        Input for theta values (for PybsmPerturber).
    theta_key : pn.widgets.Select
        Selector for the theta key (for Step/LinSpace/OneStep factories).
    start : pn.widgets.FloatInput
        Input for the start value (for Step/LinSpace factories).
    stop : pn.widgets.FloatInput
        Input for the stop value (for Step/LinSpace factories).
    step : pn.widgets.FloatInput or pn.widgets.IntInput
        Input for the step value (for Step/LinSpace factories).
    to_int : pn.widgets.Checkbox
        Checkbox to indicate if the output should be integer (for Step factory).
    theta_value : pn.widgets.FloatInput
        Input for the theta value (for OneStep factory).
    altitude_provider : pn.widgets.FloatInput
        Input for altitude (PyBSM).
    ground_range_provider : pn.widgets.FloatInput
        Input for ground range (PyBSM).
    scenario_name_provider : pn.widgets.TextInput
        Input for scenario name (PyBSM).
    ihaze_provider : pn.widgets.Select
        Selector for IHAZE value (PyBSM).
    aircraft_speed_provider : pn.widgets.FloatInput
        Input for aircraft speed (PyBSM).
    target_reflectance_provider : pn.widgets.FloatInput
        Input for target reflectance (PyBSM).
    target_temperature_provider : pn.widgets.FloatInput
        Input for target temperature (PyBSM).
    background_reflectance_provider : pn.widgets.FloatInput
        Input for background reflectance (PyBSM).
    background_temperature_provider : pn.widgets.FloatInput
        Input for background temperature (PyBSM).
    ha_windspeed_provider : pn.widgets.FloatInput
        Input for high altitude windspeed (PyBSM).
    cn2at1m_provider : pn.widgets.FloatInput
        Input for refractive index structure parameter (PyBSM).
    d_provider : pn.widgets.FloatInput
        Input for effective aperture diameter (PyBSM).
    f_provider : pn.widgets.FloatInput
        Input for focal length (PyBSM).
    sensor_name_provider : pn.widgets.TextInput
        Input for sensor name (PyBSM).
    px_provider : pn.widgets.FloatInput
        Input for detector pitch (PyBSM).
    opt_trans_wavelengths_provider : pn.widgets.LiteralInput
        Input for optical transmission wavelengths (PyBSM).
    optics_transmission_provider : pn.widgets.LiteralInput
        Input for optics transmission (PyBSM).
    eta_provider : pn.widgets.FloatInput
        Input for relative linear obscuration (PyBSM).
    int_time_provider : pn.widgets.FloatInput
        Input for integration time (PyBSM).
    dark_current_provider : pn.widgets.FloatInput
        Input for detector dark current (PyBSM).
    read_noise_provider : pn.widgets.FloatInput
        Input for RMS read noise (PyBSM).
    max_N_provider : pn.widgets.FloatInput
        Input for maximum ADC level (PyBSM).
    bit_depth_provider : pn.widgets.FloatInput
        Input for bit depth (PyBSM).
    max_well_fill_provider : pn.widgets.FloatInput
        Input for max well fill (PyBSM).
    s_x_provider : pn.widgets.FloatInput
        Input for RMS jitter amplitude X (PyBSM).
    s_y_provider : pn.widgets.FloatInput
        Input for RMS jitter amplitude Y (PyBSM).
    da_x_provider : pn.widgets.FloatInput
        Input for angular drift rate X (PyBSM).
    da_y_provider : pn.widgets.FloatInput
        Input for angular drift rate Y (PyBSM).
    qe_provider : pn.widgets.LiteralInput
        Input for quantum efficiency (PyBSM).
    qewavelengths_provider : pn.widgets.LiteralInput
        Input for quantum efficiency wavelengths (PyBSM).

    """

    title = param.String(default="Configure Natural Robustness Testing")
    next_parameter = param.Selector(
        default="Configure XAITKOD", objects=["Configure XAITKOD", "Configure XAITKIC", "ModelEvaluationTestbed"]
    )

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        from nrtk.interfaces.perturb_image import PerturbImage

        self.add_button = pn.widgets.Button(
            name="Add Test Stage", button_type="primary"
        )  # declare here since its used in pn.depends
        self.clear_button = pn.widgets.Button(
            name="Clear Test Stages", button_type="primary"
        )  # declare here since its used in pn.depends
        avail_perturbers = dict(
            sorted({pert_impl.__name__: pert_impl for pert_impl in PerturbImage.get_impls()}.items())
        )
        self.perturber_select = pn.widgets.Select(
            name="Perturber",
            options=avail_perturbers,
            value=avail_perturbers[next(iter(avail_perturbers))],
        )

        super().__init__(styles, **params)
        self.add_button.stylesheets = [self.styles.css_button]
        self.clear_button.stylesheets = [self.styles.css_button]
        self.perturber_select.stylesheets = [self.styles.widget_stylesheet]

        self.add_button.on_click(self.add_test_stage_callback)
        self.clear_button.on_click(self.clear_test_stage_callback)

        self.test_perturber_button = pn.widgets.Button(
            name="Test Perturber Settings", button_type="primary", stylesheets=[self.styles.css_button]
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
                    header_color=self.styles.color_gray_900,
                )
            ]

        self.all_widgets = [
            pn.Card(
                self.add_perturber_config_widget(),
                title="Factory Configuration",
                header_color=self.styles.color_gray_900,
                width=self.left_column_width,
            )
        ]

        self.test_stages = []
        self.finished_factory_display = []

        self.perturber_select.link(self.perturber_select, callbacks={"value": single_perturber_callback})

    def __extract_aug_img(self, img: NDArray | tuple[NDArray, Any]) -> NDArray[Any]:
        """Extract augmented image from possible tuple output.

        Returned augmented images can be as NDArray or tuple of NDArray.
        If tuple, the first element is the augmented image and the second is the dtype.

        Parameters
        ----------
        img : NDArray | tuple[NDArray, Any]
            The input image, which can be a NumPy array or a tuple containing
            the array and its dtype.

        Returns
        -------
        NDArray[Any]
            The extracted augmented image as a NumPy array.
        """
        if isinstance(img, tuple):
            return img[0]
        return img

    def add_perturber_config_widget(self) -> pn.Column:
        """Add perturber factory config widget.

        Returns
        -------
        pn.Column
            A Panel Column containing the configuration widgets for the
            selected perturber.
        """
        from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

        pert_impl = self.perturber_select.value

        factory_options = {}
        self.name_input = pn.widgets.TextInput(name="Stage Name", placeholder="Enter a name here...")
        self.name_input.stylesheets = [self.styles.widget_stylesheet]

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
        self.factory_selector.stylesheets = [self.styles.widget_stylesheet]

        if pert_impl.__name__ == "PybsmPerturber":
            self.theta_keys_input = pn.widgets.LiteralInput(name="Theta Keys", type=list, value=[])
            self.theta_keys_input.stylesheets = [self.styles.widget_stylesheet]
            self.thetas_input = pn.widgets.LiteralInput(name="Thetas", type=list, value=[])
            self.thetas_input.stylesheets = [self.styles.widget_stylesheet]
            return pn.Column(
                self.name_input,
                self._setup_scenario_parameters(),
                self._setup_sensor_parameters(),
                self.theta_keys_input,
                self.thetas_input,
            )

        return pn.Column(self.factory_selector, self.name_input, self.factory_config)

    @pn.depends("factory_selector.value")
    def factory_config(self) -> pn.Column:
        """Get factory configuration widgets based on selected factory.

        Returns
        -------
        pn.Column
            A Panel Column containing the specific configuration widgets for the
            selected factory type.
        """
        pert_impl = self.perturber_select.value
        factory_impl = self.factory_selector.value

        theta_keys_options = list(pert_impl.get_default_config().keys())
        # remove 'rng' options as it is invalid in skimage (v0.25.0)
        # (https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/-/issues/178)
        theta_keys_options.remove("rng") if "rng" in theta_keys_options else None

        if factory_impl.__name__ == "StepPerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=theta_keys_options)
            self.theta_key.stylesheets = [self.styles.widget_stylesheet]
            self.start = pn.widgets.FloatInput(name="Start", value=0.0)
            self.start.stylesheets = [self.styles.widget_stylesheet]
            self.stop = pn.widgets.FloatInput(name="Stop", value=1.0)
            self.stop.stylesheets = [self.styles.widget_stylesheet]
            self.step = pn.widgets.FloatInput(name="Step", value=1.0)
            self.step.stylesheets = [self.styles.widget_stylesheet]
            self.to_int = pn.widgets.Checkbox(name="Return integers", value=True)
            self.to_int.stylesheets = [self.styles.widget_stylesheet]

            return pn.Column(self.theta_key, self.start, self.stop, self.step, self.to_int)
        if factory_impl.__name__ == "LinSpacePerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=theta_keys_options)
            self.theta_key.stylesheets = [self.styles.widget_stylesheet]
            self.start = pn.widgets.FloatInput(name="Start", value=0.0)
            self.start.stylesheets = [self.styles.widget_stylesheet]
            self.stop = pn.widgets.FloatInput(name="Stop", value=1.0)
            self.stop.stylesheets = [self.styles.widget_stylesheet]
            self.step = pn.widgets.IntInput(name="Step", value=1)
            self.step.stylesheets = [self.styles.widget_stylesheet]

            return pn.Column(self.theta_key, self.start, self.stop, self.step)
        if factory_impl.__name__ == "OneStepPerturbImageFactory":
            self.theta_key = pn.widgets.Select(name="Theta Key", options=theta_keys_options)
            self.theta_key.stylesheets = [self.styles.widget_stylesheet]
            self.theta_value = pn.widgets.FloatInput(name="Theta Value", value=0.0)
            self.theta_value.stylesheets = [self.styles.widget_stylesheet]

            return pn.Column(self.theta_key, self.theta_value)
        bad_factory_text = pn.widget.StaticText(value=f"{factory_impl.__name__} is not supported")
        bad_factory_text.stylesheets = [self.styles.widget_stylesheet]
        return pn.Column(bad_factory_text)

    def _pybsm_parameter_init(self) -> None:
        """Initialize PyBSM specific parameter widgets."""
        self.altitude_provider = pn.widgets.FloatInput(name="Altitude (m)")
        self.altitude_provider.description = (
            "Sensor height above ground level in meters. The database includes the following "
            "altitude options: 2m 32.55m 75m 150m 225m 500m, 1000m to 12000m in 1000m steps, "
            "14000m to 20000m in 2000m steps, and 24500m"
        )
        self.altitude_provider.value = 75
        self.altitude_provider.stylesheets = [self.styles.widget_stylesheet]

        self.ground_range_provider = pn.widgets.FloatInput(name="Ground Range (m)")
        self.ground_range_provider.description = (
            "Distance on the ground between the target and sensor in meters. The following "
            "ground ranges are included in the database at each altitude until the ground "
            "range exceeds the distance to the spherical earth horizon: 0m 100m 500m, 1000m to "
            "20000m in 1000m steps, 22000m to 80000m in 2000m steps, and 85000m to "
            "300000m in 5000m steps."
        )
        self.ground_range_provider.value = 0
        self.ground_range_provider.stylesheets = [self.styles.widget_stylesheet]

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
        self.d_provider.stylesheets = [self.styles.widget_stylesheet]

        self.f_provider = pn.widgets.FloatInput(name="Focal Length (m)")
        self.f_provider.value = 0.014
        self.f_provider.stylesheets = [self.styles.widget_stylesheet]

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
        """Set up widgets for PyBSM scenario parameters.

        Returns
        -------
        pn.Column
            A Panel Column containing the scenario parameter widgets.
        """
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
            header_color=self.styles.color_gray_900,
        )

        for widget in additional_params.objects:
            widget.stylesheets = [self.styles.widget_stylesheet]

        return pn.Column(self.altitude_provider, self.ground_range_provider, additional_params)

    def _setup_sensor_parameters(self) -> pn.Column:
        """Set up widgets for PyBSM sensor parameters.

        Returns
        -------
        pn.Column
            A Panel Column containing the sensor parameter widgets.
        """
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
            header_color=self.styles.color_gray_900,
        )

        additional_params.header_color = self.styles.color_gray_900

        for widget in additional_params.objects:
            widget.stylesheets = [self.styles.widget_stylesheet]

        return pn.Column(self.d_provider, self.f_provider, additional_params)

    def add_test_stage_callback(self, _event: object) -> None:
        """Callback for the 'Add Test Stage' button.

        Creates a new perturber configuration based on current widget values
        and adds it to the list of test stages.

        Parameters
        ----------
        _event : object
            The event object from the button click (unused).
        """
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
              color: {self.styles.color_gray_900};
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
        """Callback for the 'Clear Test Stages' button.

        Clears all stored test stage configurations.

        Parameters
        ----------
        _event : object
            The event object from the button click (unused).
        """
        self.test_stages = []
        self.finished_factory_display = []

    def create_original_plot(self) -> plt.Figure:
        """Create a Matplotlib figure of the base image.

        Returns
        -------
        plt.Figure
            A Matplotlib figure object displaying the original image.
        """
        # Configure path to datasets
        img = np.asarray(Image.open(EXAMPLE_IMG))
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_title("Original Image")
        ax.axis("off")
        ax.imshow(img)
        plt.close()
        return fig

    def test_perturber_button_callback(self, _event: object) -> None:  # noqa PT019
        """Callback for the 'Test Perturber Settings' button.

        Runs the configured perturber factory on a sample image and displays
        the original and augmented images.

        Parameters
        ----------
        _event : object
            The event object from the button click (unused).
        """
        from nrtk.interfaces.perturb_image_factory import PerturbImageFactory

        self.status_source.emit("Perturbing...")
        factory_config = self.build_factory_json()
        if len(factory_config) == 0:
            return
        factory = from_config_dict(factory_config, PerturbImageFactory.get_impls())
        perturber = factory[len(factory) - 1]
        img = np.asarray(Image.open(EXAMPLE_IMG))
        perturbed_img = perturber(image=img, additional_params={"img_gsd": 3.19 / 160})
        perturbed_img = self.__extract_aug_img(perturbed_img)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis("off")
        ax.imshow(perturbed_img)
        ax.set_title("Augmented Image")
        plt.close()
        self.augmented_plot.object = fig
        self.augmented_plot.visible = True
        self.status_source.emit("Finished Perturbing")

    def _parse_pybsm_factory_config(self) -> tuple[dict, dict]:
        """Parse PyBSM factory configuration from widgets.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing two dictionaries:
            - scenario_config: Configuration for the PyBSM scenario.
            - sensor_config: Configuration for the PyBSM sensor.
        """
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
        """Collect all values from current widgets to build factory JSON.

        Returns
        -------
        dict
            A dictionary representing the configuration for the selected
            perturber factory. Returns an empty dictionary if there's an error
            or if the factory is not supported.
        """
        from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
        from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor

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
            if thetas is None or theta_keys is None:
                self.status_source.emit("Thetas and theta keys cannot be empty")
                return {}
            if len(thetas) != len(theta_keys):
                self.status_source.emit("Thetas and theta keys are different lengths")
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
            self.status_source.emit(f"{factory_impl.__name__} is not supported")
            return {}

        return {
            "type": f"{factory_impl.__module__}.{factory_impl.__name__}",
            f"{factory_impl.__module__}.{factory_impl.__name__}": output_config,
        }

    @pn.depends("add_button.clicks", "clear_button.clicks", "perturber_select.value")
    def view_sweep_params(self) -> pn.Row:
        """Update sweep parameters view.

        Updates when the 'Add Test Stage' or 'Clear Test Stages' buttons are
        clicked, or when the perturber selection changes.

        Returns
        -------
        pn.Row
            A Panel Row containing the sweep parameter configuration widgets.
        """
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
        """View for displaying original and augmented image plots.

        Returns
        -------
        pn.Column
            A Panel Column containing the original and augmented image plots.
        """
        return pn.Column(self.original_plot, self.augmented_plot)

    def view_logo(self) -> pn.pane.Image:
        """View NRTK logo.

        Returns
        -------
        pn.pane.Image
            A Panel Image pane displaying the NRTK logo.
        """
        return pn.pane.Image(
            str(NRTK_LOGO),
            width=140,
            styles={"display": "block", "float": "right", "background-color": "rgba(255, 255, 255, 1.0)"},
            stylesheets=[self.styles.text_color_styling],
        )

    def panel(self) -> pn.Column:
        """High-level view of the full application panel.

        Returns
        -------
        pn.Column
            A Panel Column representing the entire application layout.
        """
        left_column = pn.Column(
            self.perturber_select,
            pn.Spacer(height=10),  # added spacer for visual separation
            self.view_sweep_params,
            width=self.left_column_width,
        )
        right_column = pn.Column(self.view_plots)
        return pn.Column(
            self.view_header,
            pn.Row(self.view_title, pn.layout.HSpacer(), self.view_logo),
            pn.Row(
                left_column,
                right_column,
            ),
            pn.Row(pn.layout.HSpacer(), self.test_perturber_button),
            self.view_status_bar,
            pn.Spacer(height=24),
            pn.Row(pn.HSpacer(), self.next_button),
            width=self.styles.app_width,
            styles={"background": self.styles.color_main_bg},
        )
