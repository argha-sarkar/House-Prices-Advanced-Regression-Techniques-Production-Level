import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.save_plots import save_plot


def plot_feature_importance(
    model,
    feature_names,
):

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        importance["Feature"][:20],
        importance["Importance"][:20],
    )

    plt.title("Top Feature Importance")

    plt.gca().invert_yaxis()

    save_plot("feature_importance.png")

    return importance
