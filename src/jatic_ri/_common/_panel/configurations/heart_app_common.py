"""Module for configuring attack settings in HeartApp."""

from typing import Any

import panel as pn
import param

# Local imports
from jatic_ri._common._panel.configurations.base_app import DEFAULT_STYLING, AppStyling, BaseApp

HEART_DOCUMENTATION_LINK = "https://heart-library.readthedocs.io/en/latest/"
ATTACK_DOCUMENTATION_LINK = "https://heart-library.readthedocs.io/en/latest/reference_materials/attack_cards/index.html"
PATCH_PARAM_DOCUMENTATION_LINK = "https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#art.attacks.evasion.AdversarialPatch"
PGD_PARAM_DOCUMENTATION_LINK = "https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/evasion.html#art.attacks.evasion.ProjectedGradientDescent"


class HeartBaseApp(BaseApp):
    """Attack Configuration UI formatted to match HeartApp's structure and styling."""

    attack_stages: list[dict[str, Any]]
    output_test_stages: param.Dict = param.Dict({})
    finished_factory_display: pn.Column  # Stores dynamically updated stage list
    patch_attack_config: param.Boolean = param.Boolean(default=False)
    pgd_attack_config: param.Boolean = param.Boolean(default=False)
    strong_attack_config: param.Boolean = param.Boolean(default=False)
    weak_attack_config: param.Boolean = param.Boolean(default=False)
    heart_link: pn.pane.Matplotlib
    # Selection Options (Now using Select widgets)
    attack_type: param.String = param.String(default=None)  # Parameter that sets
    attack_strength: param.String = param.String(default=None)
    _previous_attack: param.String = param.String(default="")
    _previous_strength: param.String = param.String(default="")

    def __init__(self, styles: AppStyling = DEFAULT_STYLING, **params: dict[str, object]) -> None:
        super().__init__(styles, **params)

        # Action Buttons
        self.add_button = pn.widgets.Button(
            name="Add Stage", button_type="primary", stylesheets=[self.styles.css_button]
        )
        self.clear_button = pn.widgets.Button(
            name="Clear Stages", button_type="primary", stylesheets=[self.styles.css_button]
        )
        # Display area for configured test stages
        self.attack_stages = []
        self.finished_factory_display = pn.Column()

        # Button Actions
        self.add_button.on_click(self.add_test_stage_callback)
        self.clear_button.on_click(self.clear_test_stage_callback)
        self.heart_link = self.create_heart_documentation()

    @param.depends("patch_attack_config", "pgd_attack_config", watch=True)
    def enforce_patch_attack_type(self) -> None:
        """Ensure only one attack type is selected at a time."""
        if self.patch_attack_config and self.pgd_attack_config:
            # Deselect the other attack type
            if self._previous_attack != "pgd_attack_config":
                self.patch_attack_config = False
            else:
                self.pgd_attack_config = False
        # Update the previous selection
        if self.patch_attack_config:
            self._previous_attack = "patch_attack_config"
        elif self.pgd_attack_config:
            self._previous_attack = "pgd_attack_config"
        else:
            self._previous_attack = None

    @param.depends("strong_attack_config", "weak_attack_config", watch=True)
    def enforce_single_attack_strength(self) -> None:
        """Ensure only one attack strength is selected at a time."""
        if self.strong_attack_config and self.weak_attack_config:
            # Deselect the other attack type
            if self._previous_strength != "weak_attack_config":
                self.strong_attack_config = False
            else:
                self.weak_attack_config = False
        # Update the previous selection
        if self.strong_attack_config:
            self._previous_strength = "strong_attack_config"
        elif self.weak_attack_config:
            self._previous_strength = "weak_attack_config"
        else:
            self._previous_strength = None

    def add_test_stage_to_json(self, _event: object = None) -> None:
        """Adds the test stage to the JSON file"""
        attack_stage = {
            "TYPE": "HeartTestStage",
            "CONFIG": {
                "attack_type": self.attack_type,
                "parameters": self.attack_strength,
            },
        }
        self.attack_stages.append(attack_stage)

    def create_heart_documentation(self, _target: object = None, _event: object = None) -> pn.pane.Markdown:
        """Create a pane with a documentation link"""
        # Create a Markdown pane to display the link with inline HTML styling
        return pn.pane.Markdown(
            f'<a href="{HEART_DOCUMENTATION_LINK}" target="_blank" '
            f'style="color: black; font-size: 10px;">'
            "HEART-Library Documentation</a>",
        )

    def _generate_checkbox_subsection(
        self, checkbox: pn.widgets.Checkbox, label: str, description: str, section_height: int = 80
    ) -> pn.Row:
        """Construct a viewable object for the checkbox subsections.
        Adjusted spacing and ensured text does not overlap.
        """

        return pn.Row(
            pn.Spacer(width=24),
            pn.Row(
                checkbox,
                pn.Spacer(width=4),  # Spacing between checkbox and text
                pn.Column(
                    pn.pane.Markdown(
                        label,
                        styles={
                            **self.styles.style_text_subtitle,
                            "line-height": "1.5",
                            "overflow-wrap": "break-word",
                            "max-width": "600px",
                        },
                        stylesheets=[self.styles.css_paragraph],
                    ),
                    pn.Spacer(height=2),  # Space between label and description
                    pn.pane.Markdown(
                        description,
                        styles={
                            **self.styles.style_text_body2,
                            "line-height": "1.5",
                            "overflow-wrap": "break-word",
                            "max-width": "550px",
                        },
                        stylesheets=[self.styles.css_paragraph],
                    ),
                ),
                width=649,
                height=section_height,  # Ensuring sufficient space
                styles={**self.styles.style_border, "padding": "8px"},  # Padding for better spacing
            ),
        )

    def add_attack_type_tools(self) -> pn.Column:
        """Add UI tools for selecting attack types."""

        patch_attack_checkbox = pn.widgets.Checkbox.from_param(
            self.param.patch_attack_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )
        pgd_attack_checkbox = pn.widgets.Checkbox.from_param(
            self.param.pgd_attack_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )

        return pn.Column(
            pn.Spacer(height=24),
            self._generate_checkbox_subsection(
                patch_attack_checkbox,
                "Patch Attack",
                "Patch attacks alter images by adding objects that cause "
                "misclassification or detection failure, working across multiple images both digitally "
                "and physically.",
                section_height=84,  # Increased height for better spacing
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                pgd_attack_checkbox,
                "Projected Gradient Descent (PGD)",
                "PGD attacks create subtle image "
                "perturbations that exploit model gradients to mislead classification or detection, "
                "requiring full access to the  model’s architecture and parameters.",
                section_height=84,  # Ensuring consistent spacing
            ),
            pn.Spacer(height=24),
            styles={
                "background-color": self.styles.color_white,
                "border-color": self.styles.color_border,
                "border-width": "thin",
                "border-style": "solid",
                "border-radius": "8px",
                "padding": "12px",  # Added padding to avoid text touching borders
            },
            width=697,
        )

    def add_attack_strength_tools(self) -> pn.Column:
        """Add UI tools for selecting attack types."""

        strong_attack_checkbox = pn.widgets.Checkbox.from_param(
            self.param.strong_attack_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )
        weak_attack_checkbox = pn.widgets.Checkbox.from_param(
            self.param.weak_attack_config,
            name="",
            stylesheets=[self.styles.css_checkbox],
        )

        return pn.Column(
            pn.Spacer(height=24),
            self._generate_checkbox_subsection(
                strong_attack_checkbox,
                "Stronger Attack",
                "The attack will be run for a greater number of optimization iterations, "
                "improving the likelihood of the attack disrupting model performance.",
                section_height=84,
            ),
            pn.Spacer(height=12),
            self._generate_checkbox_subsection(
                weak_attack_checkbox,
                "Weaker Attack",
                "The attack will be run for fewer optimization iterations, decreasing "
                "the likelihood of the attack disrupting model performance.",
                section_height=84,
            ),
            pn.Spacer(height=24),
            styles={
                "background-color": self.styles.color_white,
                "border-color": self.styles.color_border,
                "border-width": "thin",
                "border-style": "solid",
                "border-radius": "8px",
            },
            width=697,
        )

    def add_test_stage_callback(self, _event: object) -> None:
        """Adds a test stage and immediately updates the display."""
        self.attack_type = None
        if self.patch_attack_config:
            self.attack_type = "Patch Attack"
        elif self.pgd_attack_config:
            self.attack_type = "PGD"

        self.attack_strength = None
        if self.strong_attack_config:
            self.attack_strength = "Stronger Attack"
        elif self.weak_attack_config:
            self.attack_strength = "Weaker Attack"

        if not self.attack_type or not self.attack_strength:
            return  # Ensure a selection is made before adding

        self.add_test_stage_to_json()
        # Update display with a new Markdown block showing added stage
        self.finished_factory_display.append(
            pn.pane.Markdown(
                f"""
            <style>
            * {{
              color: {self.styles.color_gray_900};
            }}
            </style>
            * Attack Type: {self.attack_type}
            * Parameters: {self.attack_strength}
            """,
            ),
        )

    def clear_test_stage_callback(self, _event: object) -> None:
        """Clears all test stages and refreshes the display."""
        self.attack_stages = []
        self.finished_factory_display.clear()  # Clears the display area

    def _run_export(self) -> None:
        """This function collects all configurations in a dictionary object
        that is shared across app pages."""
        for index, stage in enumerate(self.attack_stages):
            self.output_test_stages[f"heart-{index}"] = stage

    def panel(self) -> pn.Column:
        """Returns the UI with proper formatting."""
        # if isinstance(self, HeartICApp) {

        # }
        # Title & Description
        formatted_task = " ".join(word.capitalize() for word in self.task.split("_"))

        title_row = pn.Column(
            pn.pane.Markdown(
                f"Heart Model {formatted_task}",
                styles=self.styles.style_text_h2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "Setup your model evaluation configuration to include tools from IBMs Heart Library:",
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
            pn.pane.Markdown(
                "Once complete, you will be able to download the configuration file to "
                "test your models and datasets",
                styles=self.styles.style_text_body2,
                stylesheets=[self.styles.css_paragraph],
            ),
        )

        # Attack Type Selection Section
        attack_selection_row = pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Select Attack Type",
                    styles=self.styles.style_text_h3,
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.pane.Markdown(
                    "Choose the attack type to evaluate your model's robustness.",
                    styles=self.styles.style_text_body2,
                    stylesheets=[self.styles.css_paragraph],
                    width=395,
                ),
                pn.pane.HTML(
                    f'<a href="{ATTACK_DOCUMENTATION_LINK}" ' f'target="_blank">HEART Attack Documentation</a>',
                    styles={
                        **self.styles.style_text_subtitle,
                        "line-height": "1.5",
                        "overflow-wrap": "break-word",
                        "max-width": "600px",
                        "color": "black",
                        "font-size": "10px",
                    },
                    stylesheets=[self.styles.css_paragraph],
                ),
            ),
            pn.Spacer(width=124),
            self.add_attack_type_tools,
        )

        # Parameters Selection Section
        param_selection_row = pn.Row(
            pn.Column(
                pn.pane.Markdown(
                    "Select Parameters",
                    styles=self.styles.style_text_h3,
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.pane.Markdown(
                    "Choose the strength of the attack, either weak or strong.",
                    styles=self.styles.style_text_body2,
                    stylesheets=[self.styles.css_paragraph],
                    width=395,
                ),
                pn.pane.Markdown(
                    "Note: A rigorous adversarial evaluation should include an adaptive "
                    "strategy by adjusting and examining effects across attack parameters.",
                    styles={**self.styles.style_text_subtitle, "font-style": "italic", "font-size": "11px"},
                    stylesheets=[self.styles.css_paragraph],
                    width=395,
                ),
                pn.pane.HTML(
                    f'<a href="{PATCH_PARAM_DOCUMENTATION_LINK}" target="_blank">'
                    f"Patch Attack Parameters Documentation</a>",
                    styles={
                        **self.styles.style_text_subtitle,
                        "line-height": "1.5",
                        "overflow-wrap": "break-word",
                        "max-width": "600px",
                        "color": "black",
                        "font-size": "10px",
                    },
                    stylesheets=[self.styles.css_paragraph],
                ),
                pn.pane.HTML(
                    f'<a href="{PGD_PARAM_DOCUMENTATION_LINK}" target="_blank">'
                    f"PGD Attack Parameters Documentation</a>",
                    styles={
                        **self.styles.style_text_subtitle,
                        "line-height": "1.5",
                        "overflow-wrap": "break-word",
                        "max-width": "600px",
                        "color": "black",
                        "font-size": "10px",
                    },
                    stylesheets=[self.styles.css_paragraph],
                ),
            ),
            pn.Spacer(width=124),
            self.add_attack_strength_tools,
        )

        # Action Buttons
        buttons_row = pn.Row(
            self.add_button,
            self.clear_button,
            sizing_mode="stretch_width",
        )

        # Test Stage Display (Updates Dynamically)
        results_card = pn.Card(
            self.finished_factory_display,  # This updates dynamically
            title="Configured Attack Stages",
            collapsed=True,
        )

        heart_documentation = pn.Column(self.heart_link)

        return pn.Column(
            self.view_header,
            pn.Row(
                pn.Spacer(width=12),
                pn.Column(
                    title_row,
                    pn.Spacer(height=24),
                    self.horizontal_line(),
                    pn.Spacer(height=24),
                    attack_selection_row,
                    pn.Spacer(height=24),
                    self.horizontal_line(),
                    pn.Spacer(height=24),
                    param_selection_row,
                    pn.Spacer(height=24),
                    self.horizontal_line(),
                    pn.Spacer(height=24),
                    buttons_row,
                    pn.Spacer(height=24),
                    results_card,
                ),
                pn.Spacer(width=24),
            ),
            pn.Spacer(width=100),
            pn.Row(heart_documentation),
            styles={"background": self.styles.color_main_bg},
            width=self.styles.app_width,
        )
