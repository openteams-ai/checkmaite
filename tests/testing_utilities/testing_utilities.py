"""Module containing utility functions that can aid in testing."""
import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from pyspark.sql import DataFrame

EXAMPLE_DATA_DIR = Path(os.path.abspath(__file__)).parent / "example_data"


def assert_spark_dataframes_equal(
    actual: DataFrame,
    expected: pd.DataFrame,
    orderby: Optional[Union[str, list[str]]] = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Check whether a SparkDataFrame has the same values as a pandas.DataFrame.

    :param actual: Spark dataframe
    :param expected: The expected dataframe as a Pandas dataframe
    :param orderby: Optional param used to order columns in a dataframe
    """
    expected_sorted = (
        expected.sort_values(orderby).reset_index(drop=True)
        if orderby is not None
        else expected.reset_index(drop=True)
    )
    actual_as_pandas: pd.DataFrame = actual.toPandas()
    actual_pandas_sorted = (
        actual_as_pandas.sort_values(orderby).reset_index(drop=True)
        if orderby
        else actual_as_pandas.reset_index(drop=True)
    )
    # check_dtype=False: to ignore differences between, e.g., np.int64 vs np.int32
    # check_like=True: to ignore column order change from `toPandas`
    return pd.testing.assert_frame_equal(
        actual_pandas_sorted,
        expected_sorted,
        check_dtype=False,
        **kwargs,
    )
