from optuna import Trial


def xgboost_space(
    trial: Trial,
):

    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            200,
            3000,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            12,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0,
        ),
    }
