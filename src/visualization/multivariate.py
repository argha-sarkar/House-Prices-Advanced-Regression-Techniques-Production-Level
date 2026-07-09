import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualization.save_plots import save_plot


class MultivariateAnalysis:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe

    def correlation_heatmap(self):

        corr = self.df.select_dtypes("number").corr()

        plt.figure(figsize=(18, 15))

        sns.heatmap(corr, cmap="coolwarm", center=0)

        plt.title("Correlation Matrix")

        save_plot("correlation_heatmap.png")
