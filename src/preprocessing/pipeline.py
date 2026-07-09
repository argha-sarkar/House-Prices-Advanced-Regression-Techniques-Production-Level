import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.transformers import HouseFeatureEngineer

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)


# ... (your imports and pipelines remain the same)


def build_preprocessor(dataframe: pd.DataFrame):
    # 1. Identify features immediately
    numeric_features = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_features = dataframe.select_dtypes(exclude=np.number).columns.tolist()

    # 2. Cleanup
    if "SalePrice" in numeric_features:
        numeric_features.remove("SalePrice")

    # 3. Use the variables directly here
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor


def get_full_pipeline(dataframe: pd.DataFrame):
    """Factory function to build the full pipeline."""
    preprocessor = build_preprocessor(dataframe)

    return Pipeline(
        steps=[
            ("feature_engineering", HouseFeatureEngineer()),
            ("preprocessing", preprocessor),
        ]
    )
