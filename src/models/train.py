import pandas as pd

from src.models.evaluate import evaluate_model
from src.models.model_factory import get_models
from src.models.save_model import save_model


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
):

    results = []

    best_model = None

    best_rmse = float("inf")

    for name, model in get_models().items():

        print(f"Training {name}")

        model.fit(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["Model"] = name

        results.append(metrics)

        if metrics["RMSE"] < best_rmse:

            best_rmse = metrics["RMSE"]

            best_model = model

            save_model(
                model,
                "best_model.pkl",
            )

    results = pd.DataFrame(results)

    return pd.DataFrame(results).sort_values("RMSE"), best_model
