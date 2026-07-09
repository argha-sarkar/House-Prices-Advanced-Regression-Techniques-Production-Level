import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from src.visualization.save_plots import save_plot


def pdp_plot(
    model,
    X,
    feature,
):

    PartialDependenceDisplay.from_estimator(
        model,
        X,
        [feature],
    )

    save_plot(f"pdp_{feature}.png")
