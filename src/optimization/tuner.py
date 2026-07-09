import optuna

from src.optimization.objective import objective


def tune_model(
    X,
    y,
):

    study = optuna.create_study(
        direction="minimize",
        study_name="house_prices",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
        ),
        n_trials=100,
        show_progress_bar=True,
    )

    return study
