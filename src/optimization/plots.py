from optuna.visualization import (plot_optimization_history,
                                  plot_param_importances)


def save_plots(
    study,
):

    fig = plot_optimization_history(study)

    fig.write_html("reports/optimization_history.html")

    fig = plot_param_importances(study)

    fig.write_html("reports/parameter_importance.html")
