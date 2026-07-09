import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.optimization.search_space import xgboost_space
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.transformers import HouseFeatureEngineer


def objective(
    trial,
    X,
    y,
):

    params = xgboost_space(trial)

    model = XGBRegressor(
        **params,
        random_state=42,
        verbosity=0,
    )

    pipeline = Pipeline(
        steps=[
            (
                "feature_engineering",
                HouseFeatureEngineer(),
            ),
            (
                "preprocessing",
                build_preprocessor(X),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    score = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    return -np.mean(score)
