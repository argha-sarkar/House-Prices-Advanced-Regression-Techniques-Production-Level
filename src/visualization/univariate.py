import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.save_plots import save_plot


class UnivariateAnalysis:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def histogram(self, column: str):

        plt.figure()

        sns.histplot(self.df[column], kde=True)

        plt.title(f"Distribution of {column}")

        save_plot(f"{column}_histogram.png")

    def boxplot(self, column: str):

        plt.figure()

        sns.boxplot(x=self.df[column])

        plt.title(f"Boxplot of {column}")

        save_plot(f"{column}_boxplot.png")

    def violin(self, column: str):

        plt.figure()

        sns.violinplot(x=self.df[column])

        plt.title(f"Violin Plot of {column}")

        save_plot(f"{column}_violin.png")
