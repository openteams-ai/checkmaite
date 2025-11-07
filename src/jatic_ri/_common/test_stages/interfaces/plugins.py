"""Plugins to be used for image classification and object detection implementations of
TestStage classes. Implementations may use one or more plugins depending on the application.
"""


class ThresholdPlugin:
    """TestStage Plugin for loading a threshold.

    Attributes
    ----------
    threshold : float
        The threshold value.
    """

    threshold: float

    def load_threshold(self, threshold: float) -> None:
        """Set threshold for the test.

        Parameters
        ----------
        threshold : float
            The threshold value to set.
        """
        self.threshold = threshold
