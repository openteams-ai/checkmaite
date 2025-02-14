"""
This module contains the BaseApp class, which serves as the base class for all individual configuration pages.
It provides common methods and visualization elements that can be utilized by the individual pages.
"""

import panel as pn
import param

from jatic_ri import PACKAGE_DIR

JATIC_LOGO_PATH = PACKAGE_DIR.joinpath(
    "_sample_imgs",
    "JATIC_Logo_Acronym_Spelled_Out_RGB_white_type.svg",
)

pn.extension("tabulator")


class BaseApp(param.Parameterized):
    """Base class for all the individual configuration pages.
    This class holds common methods and visualization elements.

    The individual pages should utilize:
    * a custom title
    * call `self.view_status_bar` for the status bar visualiation
      * to change the text on the status bar, use `self.status_text`
    * call `self.view_title` for the title visualization
    * apply dropdown/float/int widget input styling like this `pn.widgets.Select.from_param(self.param.model,
      width=self.widget_width, name='Model', stylesheets=[self.widget_stylesheet])`
    * apply width to the high level viewable (the outermost element returned from the `panel` method)
       - width=self.page_width
    * apply the navy background to the overall page `styles=dict(background=self.color_main_bg)`
    * apply the button stylesheets, button_type must be set to "primary"
    * visualize the export button and implement the _run_export method to run/collect metrics, etc
    * remove pn.extension from app.py files

    NOTES: # TO DO - finish renaming non-compliant attributes
    * self.color_*: reserved for HEX color codes
    * self.style_*: css styling applied via `styles` input to Panel widgets, e.g. `styles=self.style_foo`
    * self.css_*: css styling applied via `css` input to Panel widgets, e.g. `css=[self.css_foo]`
    """

    page_width: int = param.Integer(800)
    page_height: int = param.Integer(800)
    title_font_size: int = param.Integer(default=24)
    status_text: str = param.String("Waiting for input...")
    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    summary_text_size: int = param.Integer(default=18)
    export_button: pn.widgets.Button

    ##################################################
    # Pipeline specific parameters
    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})
    # flag for local deployment
    local = param.Boolean()
    ##################################################

    ##################################################
    # individual apps should override these parameters
    title: str = param.String(default="Base class title: please update")
    ##################################################

    def __init__(self, **params: dict[str, object]) -> None:
        super().__init__(**params)
        self.app_width = 1280
        # style guide
        self.color_blue_900 = "#001B4D"  # blue-900
        self.color_blue_800 = "#0F388A"  # blue-800
        self.color_blue_700 = "#1550C1"  # blue-700
        self.color_blue_600 = "#195FE6"  # blue-600
        self.color_blue_500 = "#5284E5"  # blue-500
        self.color_blue_400 = "#770FEE"  # blue-400
        self.color_blue_300 = "#A3BEF5"  # blue-300
        self.color_blue_200 = "#D5E0F6"  # blue-200
        self.color_blue_100 = "#EDF2FD"  # blue-100
        self.color_white = "#FFFFFF"  # pure-white
        self.color_gray_900 = "#00050A"  # gray-900
        self.color_gray_800 = "#1E2C3E"  # gray-800
        self.color_gray_700 = "#415062"  # gray-700
        self.color_gray_600 = "#788BA5"  # gray-600
        self.color_gray_500 = "#BBC9DD"  # gray-500
        self.color_gray_400 = "#DDE4EE"  # gray-400
        self.color_gray_300 = "#F1F4F9"  # gray-300
        self.color_gray_200 = "#F8FAFC"  # gray-200
        self.color_main_bg = self.color_gray_200

        self.color_maintext = self.color_gray_900  # gray-900
        self.color_subtext = self.color_gray_700  # gray-700
        self.color_border = self.color_gray_500  # gray-500

        self.font_family = "'Helvetica Neue', 'Arial'"
        self.style_text_h1 = {
            "font-size": "24px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_maintext,
        }
        self.style_text_h2 = {
            "font-size": "18px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_maintext,
        }
        self.style_text_h3 = {
            "font-size": "13px",
            "font-family": self.font_family,
            "font-weight": "bold",
            "color": self.color_maintext,
        }
        self.style_text_subtitle = {
            "font-size": "12px",
            "font-family": self.font_family,
            "font-weight": "semibold",
            "color": self.color_maintext,
        }
        self.style_text_body1 = {
            "font-size": "13px",
            "font-family": self.font_family,
            "color": self.color_maintext,
        }
        self.style_text_body2 = {
            "font-size": "12px",
            "font-family": self.font_family,
            "color": self.color_subtext,
        }

        self.style_border = {
            "background-color": self.color_white,
            "border-color": self.color_border,
            "border-width": "thin",
            "border-style": "solid",
            "border-radius": "8px",
        }

        # removes paragraph margins and overrides font-family
        self.css_paragraph = """
            :host p {
              margin: 0px;
              font-family: "Helvetica Neue", "Arial";
            }
            """
        # adjust the dimensions of a checkbox widget
        self.css_checkbox = """
            input {
                height: 16px;
                width: 16px;
            }
            """
        # adjust button styling
        self.css_button = f"""
            :host(.solid) .bk-btn.bk-btn-default {{
              background-color: {self.color_blue_500};
              color: #FFFFFF;
            }}
            """
        # adjust switch toggle styling
        self.css_switch = f"""
            :host(.active) .knob {{
                background-color:{self.color_blue_500};
            }}
            :host(.active) .bar {{
                background-color: {self.color_blue_200};
            }}
            """
        self.widget_width = (
            140  # the widget construct is overriding this via css so we'll specify on the widget object instead
        )
        self.widget_height = "20px"
        self.widget_stylesheet = f"""
            :host {{
              color: {self.color_gray_700}; /* label text color */
            }}

            select:not([multiple]).bk-input, select:not([size]).bk-input {{
              height: {self.widget_height}; /* dropdown widget height */
              color: {self.color_gray_900} /* text color on value of Dropdown widgets */
            }}

            .bk-input {{
              height: {self.widget_height};  /* FloatInput widget height */
              color: {self.color_gray_900} /* text color on value of FloatInput widgets */
            }}
            """
        self.button_bgcolor = self.color_blue_500
        self.button_textcolor = self.color_white
        self.button_stylesheet = f"""
            :host(.solid) .bk-btn.bk-btn-primary {{
              background-color: {self.button_bgcolor}
            }}

            .bk-btn-primary {{
              color: {self.button_textcolor}
            }}
            """
        self.text_color_styling = f"""
            *, *:before, *:after  {{
              color: {self.color_gray_700};
            }}
            """
        self.info_button_style = f"""
            .bk-description > .bk-icon {{
              background-color: {self.color_gray_700};
            }}
            """

        # create export button and connect it to a callback method
        self.export_button = pn.widgets.Button(
            name="Export Configuration",
            button_type="primary",
            stylesheets=[self.button_stylesheet],
        )
        self.export_button.on_click(self.export_button_callback)

        self.output_path = ""

    def _run_export(self) -> None:
        """Individual implementation of export process.
        This method should populate the self.output_test_stages with
        {tool_name: test_stage_dict}. The dictionary you provide should
        be JSON serializable and should be readable by your individual
        TestStage class.
        EVERY TEAM SHOULD OVERWRITE THIS METHOD.
        """
        print("THIS SHOULD BE IMPLEMENTED IN THE HIGHER LEVEL CLASS")
        # for testing and demo purposes only:
        self.output_test_stages[self.__class__.__name__] = {"TYPE": "base app test"}

    @param.output(task=param.Selector, output_test_stages=param.Dict, local=param.Boolean)
    def output(self) -> tuple:
        """Output handler for passing variables from one pipeline page to another"""
        self._run_export()
        return self.task, self.output_test_stages, self.local

    def export_button_callback(self, _event: object) -> None:
        """Method to take the configuration and save to disk.
        DO NOT OVERWRITE THIS METHOD.
        """
        self._run_export()
        self.status_text = "Configuration saved"

    def horizontal_line(self) -> pn.pane.HTML:
        """Creates a horizontal line in an HTML element.
        This is a function because the same panel object
        cannot be visualized twice so in order to reuse
        this code snippet, we need to create a new object
        every time we need a line.
        """
        return pn.pane.HTML(
            """""",
            styles={
                "display": "block",
                "height": "1px",
                "border": "0",
                "border-top": f"1px solid {self.color_border}",
                "margin": "0em 0",
                "padding": "0",
            },
            width=self.app_width - 48,
        )

    def view_status_bar(self) -> pn.Column:
        """View of status bar. Change the text on the status bar by modifying
        the `self.status_text` variable.
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.Column(
            pn.Spacer(height=20),
            self.horizontal_line(),
            pn.Row(
                pn.pane.Markdown("Status:", styles=self.style_text_body1),
                pn.pane.Markdown(
                    self.status_text,
                    sizing_mode="stretch_width",
                    styles=self.style_text_body1,
                ),
            ),
        )

    def view_title(self) -> pn.pane.Markdown:
        """View of the app title
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.pane.Markdown(
            self.title,
            styles=self.style_text_h1,
        )

    def view_header(self) -> pn.Row:
        """View header row with JATIC logo"""
        return pn.Row(
            pn.pane.SVG(JATIC_LOGO_PATH, width=150),
            styles={"background": self.color_blue_900},
            width=self.app_width,
        )

    def panel(self) -> pn.Column:
        """Example usage of this base class
        EVERY TEAM SHOULD  OVERWRITE THIS METHOD
        """
        return pn.Column(
            self.view_header,
            self.view_title,
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_main_bg},
        )
