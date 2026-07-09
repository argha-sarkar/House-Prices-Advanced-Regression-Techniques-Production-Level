from pathlib import Path

import matplotlib.pyplot as plt

from src.paths import FIGURES_DIR


def save_plot(filename: str, dpi: int = 300) -> None:
    """
    Save the current figure.
    """

    path = FIGURES_DIR / filename

    plt.savefig(path, dpi=dpi, bbox_inches="tight")

    plt.close()
