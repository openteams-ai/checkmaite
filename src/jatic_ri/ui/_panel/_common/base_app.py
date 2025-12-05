"""
This module contains the BaseApp class, which serves as the base class for all individual configuration pages.
It provides common methods and visualization elements that can be utilized by the individual pages.
"""

from pathlib import Path

import panel as pn
import param
from pydantic import BaseModel, ConfigDict, model_validator
from streamz import Stream

JATIC_LOGO_PATH = Path(__file__).parents[3] / "assets" / "JATIC_Logo_Acronym_Spelled_Out_RGB_white_type.svg"

pn.extension("tabulator", "filedropper")


class AppStyling(BaseModel):
    """Pydantic v2 model for styling parameters.

    Attributes
    ----------
    app_width : int
        Base setting for application width.
    color_blue_900 : str
        Hex code for a specific shade of blue.
    color_blue_800 : str
        Hex code for a specific shade of blue.
    color_blue_700 : str
        Hex code for a specific shade of blue.
    color_blue_600 : str
        Hex code for a specific shade of blue.
    color_blue_500 : str
        Hex code for a specific shade of blue.
    color_blue_400 : str
        Hex code for a specific shade of blue.
    color_blue_300 : str
        Hex code for a specific shade of blue.
    color_blue_200 : str
        Hex code for a specific shade of blue.
    color_blue_100 : str
        Hex code for a specific shade of blue.
    color_white : str
        Hex code for white.
    color_gray_900 : str
        Hex code for a specific shade of gray.
    color_gray_800 : str
        Hex code for a specific shade of gray.
    color_gray_700 : str
        Hex code for a specific shade of gray.
    color_gray_600 : str
        Hex code for a specific shade of gray.
    color_gray_500 : str
        Hex code for a specific shade of gray.
    color_gray_400 : str
        Hex code for a specific shade of gray.
    color_gray_300 : str
        Hex code for a specific shade of gray.
    color_gray_200 : str
        Hex code for a specific shade of gray.
    widget_width : int
        Default width for widgets.
    widget_height : str
        Default height for widgets.
    width_input_default : int
        Default width for input widgets.
    width_subwidget_offset : int
        Offset for sub-widget width.
    color_main_bg : str | None
        Derived main background color.
    color_maintext : str | None
        Derived main text color.
    color_subtext : str | None
        Derived subtext color.
    color_border : str | None
        Derived border color.
    font_family : str | None
        Derived font family.
    style_text_h1 : dict[str, str] | None
        Derived style for H1 text.
    style_text_h2 : dict[str, str] | None
        Derived style for H2 text.
    style_text_h3 : dict[str, str] | None
        Derived style for H3 text.
    style_text_subtitle : dict[str, str] | None
        Derived style for subtitle text.
    style_text_body1 : dict[str, str] | None
        Derived style for body1 text.
    style_text_body2 : dict[str, str] | None
        Derived style for body2 text.
    style_border : dict[str, str] | None
        Derived style for borders.
    widget_stylesheet : str | None
        Derived stylesheet for widgets.
    button_bgcolor : str | None
        Derived background color for buttons.
    button_textcolor : str | None
        Derived text color for buttons.
    text_color_styling : str | None
        Derived CSS for text color styling.
    info_button_style : str | None
        Derived CSS for info button style.
    css_paragraph : str | None
        Derived CSS for paragraphs.
    css_center_text : str | None
        Derived CSS for centering text.
    css_checkbox : str | None
        Derived CSS for checkboxes.
    css_button : str | None
        Derived CSS for buttons.
    css_switch : str | None
        Derived CSS for switches.
    css_dropdown : str | None
        Derived CSS for dropdowns.
    css_config_input : str | None
        Derived CSS for config inputs.
    css_tabulator_table : str | None
        Derived CSS for tabulator tables.
    css_filedropper : str | None
        Derived CSS for file droppers.

    """

    # Allow us to set attributes in the validator
    model_config = ConfigDict(validate_assignment=False)

    # Base settings
    app_width: int = 1280

    # Color palette
    color_blue_900: str = "#001B4D"
    color_blue_800: str = "#0F388A"
    color_blue_700: str = "#1550C1"
    color_blue_600: str = "#195FE6"
    color_blue_500: str = "#5284E5"
    color_blue_400: str = "#770FEE"
    color_blue_300: str = "#A3BEF5"
    color_blue_200: str = "#D5E0F6"
    color_blue_100: str = "#EDF2FD"
    color_white: str = "#FFFFFF"
    color_gray_900: str = "#00050A"
    color_gray_800: str = "#1E2C3E"
    color_gray_700: str = "#415062"
    color_gray_600: str = "#788BA5"
    color_gray_500: str = "#BBC9DD"
    color_gray_400: str = "#DDE4EE"
    color_gray_300: str = "#F1F4F9"
    color_gray_200: str = "#F8FAFC"

    # Widget defaults
    widget_width: int = 140
    widget_height: str = "20px"
    width_input_default: int = 580
    width_subwidget_offset: int = 20

    # Derived fields — initialize to None here
    color_main_bg: str | None = None
    color_maintext: str | None = None
    color_subtext: str | None = None
    color_border: str | None = None
    font_family: str | None = None

    style_text_h1: dict[str, str] | None = None
    style_text_h2: dict[str, str] | None = None
    style_text_h3: dict[str, str] | None = None
    style_text_subtitle: dict[str, str] | None = None
    style_text_body1: dict[str, str] | None = None
    style_text_body2: dict[str, str] | None = None

    style_border: dict[str, str] | None = None

    widget_stylesheet: str | None = None
    button_bgcolor: str | None = None
    button_textcolor: str | None = None

    text_color_styling: str | None = None
    info_button_style: str | None = None

    css_paragraph: str | None = None
    css_center_text: str | None = None
    css_checkbox: str | None = None
    css_button: str | None = None
    css_switch: str | None = None
    css_dropdown: str | None = None
    css_config_input: str | None = None
    css_tabulator_table: str | None = None

    css_filedropper: str | None = None

    @model_validator(mode="after")
    def compute_derived(self):  # noqa: ANN201
        """Compute derived fields based on the base settings.

        This method is a Pydantic model validator that runs after
        initialization to populate derived styling attributes.
        """
        # alias locals for readability
        cg200 = self.color_gray_200
        cg900 = self.color_gray_900
        cg700 = self.color_gray_700
        cg500 = self.color_gray_500
        cwhite = self.color_white
        cb500 = self.color_blue_500
        cb100 = self.color_blue_100
        cb200 = self.color_blue_200
        cb400 = self.color_blue_400

        # simple mappings
        self.color_main_bg = cg200
        self.color_maintext = cg900
        self.color_subtext = cg700
        self.color_border = cg500
        self.font_family = "'Helvetica Neue', 'Arial'"

        # text styles
        ff = self.font_family
        self.style_text_h1 = {"font-size": "24px", "font-family": ff, "font-weight": "bold", "color": cg900}
        self.style_text_h2 = {"font-size": "18px", "font-family": ff, "font-weight": "bold", "color": cg900}
        self.style_text_h3 = {"font-size": "13px", "font-family": ff, "font-weight": "bold", "color": cg900}
        self.style_text_subtitle = {"font-size": "12px", "font-family": ff, "font-weight": "semibold", "color": cg900}
        self.style_text_body1 = {"font-size": "13px", "font-family": ff, "color": cg900}
        self.style_text_body2 = {"font-size": "12px", "font-family": ff, "color": cg700}

        # border style
        self.style_border = {
            "background-color": cwhite,
            "border-color": cg500,
            "border-width": "thin",
            "border-style": "solid",
            "border-radius": "8px",
        }

        # widget stylesheet
        self.widget_stylesheet = f"""
            :host {{
                color: {cg700};
            }}
            select:not([multiple]).bk-input, select:not([size]).bk-input {{
                height: {self.widget_height};
                color: {cg900};
            }}
            .bk-input {{
                height: {self.widget_height};
                color: {cg900};
            }}
            """
        # buttons
        self.button_bgcolor = cb500
        self.button_textcolor = cwhite

        # other CSS snippets...
        self.text_color_styling = f"*, *:before, *:after {{ color: {cg700}; }}"
        self.info_button_style = f".bk-description > .bk-icon {{ background-color: {cg700}; }}"

        self.css_paragraph = """
            :host p {
                margin: 0px;
                font-family: "Helvetica Neue", "Arial";
            }"""
        self.css_center_text = """
            :host p {
                text-align: center;
            }"""
        self.css_checkbox = "input { height: 16px; width: 16px; }"
        self.css_button = f"""
            :host(.solid) .bk-btn.bk-btn-default {{
                background-color: {cb500};
                color: {cwhite};
            }}"""
        self.css_switch = f"""
            :host(.active) .knob {{
                background-color: {cb500};
            }}
            :host(.active) .bar {{
                background-color: {cb200};
            }}"""
        self.css_dropdown = f"""
            label {{
                color: {cg700};
            }}
            select:not([multiple]).bk-input, select:not([size]).bk-input {{
                height: {self.widget_height};
                color: {cg900};
            }}"""
        self.css_config_input = f"""
            .bk-input {{
                color: {cg900};
            }}
            input[type='file'] {{
                height: 40px;
                border: 1px dashed;
                padding: 0;
            }}"""
        self.css_tabulator_table = f"""
            .tabulator-row.tabulator-selectable:hover {{
                background-color: {cb200} !important;
            }}
            host: .tabulator-row.tabulator-selected {{
                background-color: {cb400} !important;
            }}
            .tabulator-row {{
                background-color: {cwhite} !important;
                border: none !important;
            }}"""
        self.css_filedropper = f"""
            .bk-input.filepond--root .filepond--drop-label {{
                background-color: {cb100};
            }}"""
        return self


DEFAULT_STYLING = AppStyling()


class BaseApp(param.Parameterized):
    """Base class for all individual configuration pages.

    This class holds common methods and visualization elements. Individual
    pages should utilize:
    - a custom title
    - call `self.view_status_bar` for the status bar visualization
      - to change the text on the status bar, use `self.status_source.emit("Your text")`
    - call `self.view_header` for the header visualization (with JATIC logo)
    - call `self.view_title` for the title visualization
    - apply styles from the `self.styles` object

    Parameters
    ----------
    output_test_stages : param.Dict
        Dictionary for holding all the output configurations. This dictionary
        is passed between all the stages and is used in the final stage to
        generate the json file.
    local : param.Boolean
        Flag for local deployment.
    workflow : param.Selector
        Selector for the workflow type (e.g., "model_evaluation", "dataset_analysis").
    task : param.Selector
        Selector for the task type (e.g., "object_detection", "image_classification").
    ready : param.Boolean
        Flag indicating when the app is ready to advance to the next stage.
    testbed_config : param.Dict
        Configuration for the testbed.
    title : str
        Title of the application page. Individual apps should override this.
    styles : AppStyling
        Styling configuration object.

    Attributes
    ----------
    status_source : streamz.Stream
        Stream for status text updates.
    status_pane : pn.pane.Streamz
        Panel pane to display status messages.
    next_button : pn.widgets.Button
        Button to proceed to the next stage.
    output_dir : str
        Directory where report files will be saved.

    """

    ##################################################
    # Pipeline specific parameters
    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})
    # flag for local deployment
    local = param.Boolean(default=True)
    workflow = param.Selector(default="model_evaluation", objects=["model_evaluation", "dataset_analysis"])
    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    ready = param.Boolean(default=False)  # flag for when the app is ready to advance to next stage
    testbed_config = param.Dict(default={})
    ##################################################
    output_dir = param.Path(default=Path.cwd().joinpath("report"), check_exists=False)
    ##################################################
    # individual apps should override these parameters
    title: str = param.String(default="Base class title: please update")
    ##################################################

    def __init__(self, styles: AppStyling, **params: dict[str, object]) -> None:
        super().__init__(**params)
        self.styles = styles

        # Create a Stream that will carry text updates
        self.status_source = Stream()
        # Create a Streamz pane that will display the status messages
        self.status_pane = pn.pane.Streamz(
            self.status_source,
            always_watch=True,
            sizing_mode="stretch_width",
            styles={**self.styles.style_text_body1, "color": self.styles.color_blue_800},
            stylesheets=[self.styles.css_paragraph],
        )
        # Emit an initial message
        self.status_source.emit("Waiting for input...")

        self.next_button = pn.widgets.Button(
            name="Next",
            button_type="primary",
            stylesheets=[styles.css_button],
            width=150,
        )
        self.next_button.on_click(self._next_button_callback)

    @property
    def suffix(self) -> str:
        """Task abbreviation used for suffixing variables.

        Returns
        -------
        str
            "OD" if the task is "object_detection", "IC" otherwise.
        """
        return "OD" if self.task == "object_detection" else "IC"

    def _run_export(self) -> None:
        """Individual implementation of export process.

        This method should populate `self.output_test_stages` with
        `{tool_name: test_stage_dict}`. The dictionary provided should
        be JSON serializable and readable by the individual TestStage class.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.
        """
        # for demo purposes only:
        self.output_test_stages[self.__class__.__name__] = {"TYPE": "base app test"}
        raise NotImplementedError("THIS SHOULD BE IMPLEMENTED IN THE LOWER LEVEL CLASS")

    @param.output(task=param.Selector, output_test_stages=param.Dict, local=param.Boolean)
    def output(self) -> tuple:
        """Output handler for passing variables from one pipeline page to another.

        Returns
        -------
        tuple
            A tuple containing the task, output test stages, and local flag.
        """
        self._run_export()
        return self.task, self.output_test_stages, self.local

    def _next_button_callback(self, event: param.Event) -> None:  # noqa: ARG002
        """Callback for the Next button to advance to the next stage.

        Parameters
        ----------
        event : param.Event
            The event triggered by the button click.
        """
        self.ready = True

    def horizontal_line(self) -> pn.pane.HTML:
        """Creates a horizontal line in an HTML element.

        This is a function because the same panel object cannot be visualized
        twice. To reuse this code snippet, a new object must be created
        every time a line is needed.

        Returns
        -------
        pn.pane.HTML
            A Panel HTML pane representing a horizontal line.
        """
        return pn.pane.HTML(
            """""",
            styles={
                "display": "block",
                "height": "1px",
                "border": "0",
                "border-top": f"1px solid {self.styles.color_border}",
                "margin": "0em 0",
                "padding": "0",
            },
            width=self.styles.app_width - 48,
        )

    def view_status_bar(self) -> pn.Column:
        """View of status bar.

        Change the text on the status bar by modifying the
        `self.status_source.emit` method.

        Returns
        -------
        pn.Column
            A Panel Column containing the status bar.
        """
        return pn.Column(
            pn.Spacer(height=20),
            self.horizontal_line(),
            pn.Row(
                pn.pane.Markdown("Status:", styles=self.styles.style_text_body1),
                pn.Column(
                    pn.Spacer(height=13),
                    self.status_pane,
                ),
            ),
        )

    def view_title(self) -> pn.pane.Markdown:
        """View of the app title.

        Returns
        -------
        pn.pane.Markdown
            A Panel Markdown pane displaying the application title.
        """
        return pn.pane.Markdown(
            self.title,
            styles=self.styles.style_text_h1,
        )

    def view_header(self, display_task: bool = True) -> pn.Row:
        """View header row with JATIC logo.

        Returns
        -------
        pn.Row
            A Panel Row containing the JATIC logo.
        """
        header = pn.Row(
            pn.pane.SVG(JATIC_LOGO_PATH, width=150),
            styles={"background": self.styles.color_blue_900},
            width=self.styles.app_width,
        )

        if display_task:
            header.extend(
                [
                    pn.Spacer(sizing_mode="stretch_width"),
                    pn.pane.Markdown(
                        self.task.replace("_", " ").title(),
                        styles={
                            **(self.styles.style_text_body2 or {}),
                            "color": self.styles.color_white,
                        },
                        stylesheets=[self.styles.css_paragraph],
                        align=("center", "center"),  # (horizontal, vertical) alignment
                    ),
                    pn.Spacer(width=24),
                ]
            )
        return header
