from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.paths import FIGURES_DIR

FIGURES_DIR.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid", context="talk")

plt.rcParams["figure.figsize"] = (12, 6)

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["axes.labelsize"] = 14
