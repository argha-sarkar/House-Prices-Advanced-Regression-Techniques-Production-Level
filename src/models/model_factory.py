from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from xgboost import XGBRegressor

RANDOM_STATE = 42


def get_models():

    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            random_state=RANDOM_STATE,
        ),
        "CatBoost": CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }
