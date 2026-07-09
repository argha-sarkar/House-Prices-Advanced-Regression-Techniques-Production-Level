import numpy as np
import pandas as pd


def log_transform_target(df):

    df = df.copy()

    df["SalePrice"] = np.log1p(df["SalePrice"])

    return df


def group_rare_categories(
    dataframe: pd.DataFrame,
    column: str,
    threshold: int = 10,
) -> pd.DataFrame:
    """
    Replace infrequent categories with 'Rare'.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataframe.

    column : str
        Categorical column name.

    threshold : int
        Minimum frequency required to keep a category.
    """

    dataframe = dataframe.copy()

    counts = dataframe[column].value_counts()

    rare_categories = counts[counts < threshold].index

    dataframe[column] = dataframe[column].replace(
        rare_categories,
        "Rare",
    )

    return dataframe
