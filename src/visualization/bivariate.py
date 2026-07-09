import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.save_plots import save_plot


class BivariateAnalysis:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe

    def scatter(self, feature: str, target: str):

        plt.figure()

        sns.scatterplot(data=self.df, x=feature, y=target)

        plt.title(f"{feature} vs {target}")

        save_plot(f"{feature}_scatter.png")

    def regression(self, feature: str, target: str):

        plt.figure()

        sns.regplot(data=self.df, x=feature, y=target)

        plt.title(f"{feature} Regression")

        save_plot(f"{feature}_regression.png")
