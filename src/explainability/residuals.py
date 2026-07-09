import matplotlib.pyplot as plt

from src.visualization.save_plots import save_plot


def residual_plot(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(X_test)

    residuals = y_test - predictions

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions,
        residuals,
    )

    plt.axhline(
        0,
        linestyle="--",
    )

    plt.xlabel("Prediction")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    save_plot("residual_plot.png")
