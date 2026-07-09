from __future__ import annotations

import pandas as pd

QUALITY_MAP = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
}


def encode_quality(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Ordinal encode quality columns.

    Parameters
    ----------
    dataframe : pd.DataFrame

    column : str
    """

    dataframe = dataframe.copy()

    dataframe[column] = dataframe[column].map(QUALITY_MAP)

    return dataframe
