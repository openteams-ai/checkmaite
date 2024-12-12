"""
This module contains the BaseApp class, which serves as the base class for all individual configuration pages.
It provides common methods and visualization elements that can be utilized by the individual pages.
"""

from typing import Any

import panel as pn
import param

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
    * apply the navy background to the overall page `styles=dict(background=self.color_dark_blue)`
    * apply the button stylesheets, button_type must be set to "primary"
    * visualize the export button and implement the _run_export method to run/collect metrics, etc
    * remove pn.extension from app.py files
    """

    page_width: int = param.Integer(800)
    page_height: int = param.Integer(800)
    title_font_size: int = param.Integer(default=24)
    status_text: str = param.String("Waiting for input...")
    output_test_stages: dict[str, Any] = param.Dict({})
    task = param.Selector(default="object_detection", objects=["object_detection", "image_classification"])
    summary_text_size: int = param.Integer(default=18)
    export_button: pn.widgets.Button

    ##################################################
    # Pipeline specific parameters
    # dictionary for holding all the output configurations
    # this dictionary is passed between all the stages and
    # is used in the final stage to generate the json file
    output_test_stages = param.Dict({})
    ##################################################

    ##################################################
    # individual apps should override these parameters
    title: str = param.String(default="Base class title: please update")
    ##################################################

    def __init__(self, **params: dict[str, object]) -> None:
        super().__init__(**params)
        # style guide
        self.color_light_gray = "#fafefe"
        self.color_dark_blue = "#1d385b"
        self.color_light_blue = "#abd1f5"
        self.color_black = "#000000"

        # this adjusts the margins on the title
        self.title_stylesheet = """
:host {
  --line-height: 10px;
}

p {
  padding: 0px;
  margin: 10px;
}
"""
        self.widget_width = (
            140  # the widget construct is overriding this via css so we'll specify on the widget object instead
        )
        self.widget_height = "20px"
        self.widget_stylesheet = f"""
:host {{
  color: {self.color_light_gray}; /* label text color */
}}

select:not([multiple]).bk-input, select:not([size]).bk-input {{
  height: {self.widget_height}; /* dropdown widget height */
  color: {self.color_black} /* text color on value of Dropdown widgets */
}}

.bk-input {{
  height: {self.widget_height};  /* FloatInput widget height */
  color: {self.color_black} /* text color on value of FloatInput widgets */
}}
"""
        self.button_bgcolor = self.color_light_blue
        self.button_textcolor = self.color_black
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
  color: {self.color_light_gray};
}}
"""
        self.info_button_style = f"""
.bk-description > .bk-icon {{
  background-color: {self.color_light_gray};
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

    @param.output(task=param.Selector, output_test_stages=param.Dict)
    def output(self) -> tuple:
        """Output handler for passing variables from one pipeline page to another"""
        self._run_export()
        return self.task, self.output_test_stages

    @param.output(task=param.String)
    def _task(self) -> str:  # pragma: no cover
        return self.task

    def export_button_callback(self, _event: object) -> None:
        """Method to take the configuration and save to disk.
        DO NOT OVERWRITE THIS METHOD.
        """
        self._run_export()
        self.status_text = "Configuration saved"

    def view_status_bar(self) -> pn.Column:
        """View of status bar. Change the text on the status bar by modifying
        the `self.status_text` variable.
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.Column(
            pn.layout.Divider(),
            pn.Row(
                pn.pane.Markdown("Status:", styles={"color": f"{self.color_light_gray}"}),
                pn.pane.Markdown(
                    self.status_text,
                    sizing_mode="stretch_width",
                    styles={"color": f"{self.color_light_gray}"},
                ),
            ),
        )

    def view_title(self) -> pn.pane.Markdown:
        """View of the app title
        DO NOT OVERWRITE THIS METHOD
        """
        return pn.pane.Markdown(
            self.title,
            styles={"font-size": f"{self.title_font_size}px", "color": f"{self.color_light_gray}"},
            stylesheets=[self.title_stylesheet],
        )

    def panel(self) -> pn.Column:
        """Example usage of this base class
        EVERY TEAM SHOULD  OVERWRITE THIS METHOD
        """
        return pn.Column(
            self.view_title,
            self.view_status_bar,
            width=self.page_width,
            styles={"background": self.color_dark_blue},
        )
