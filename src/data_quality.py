from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import logger
from src.paths import METRICS_DIR


class DataQualityAnalyzer:
    """
    Analyze dataset quality and generate reports.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def basic_information(self) -> pd.DataFrame:
        """
        Returns feature information.
        """

        info = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "missing_values": self.df.isna().sum(),
                "missing_percent": (self.df.isna().mean() * 100).round(2),
                "unique_values": self.df.nunique(),
            }
        )

        return info

    def duplicate_rows(self) -> int:
        """
        Count duplicate records.
        """

        return int(self.df.duplicated().sum())

    def numerical_columns(self) -> list[str]:
        """
        Return numerical columns.
        """

        return self.df.select_dtypes(include=np.number).columns.tolist()

    def categorical_columns(self) -> list[str]:
        """
        Return categorical columns.
        """

        return self.df.select_dtypes(exclude=np.number).columns.tolist()

    def missing_report(self) -> pd.DataFrame:

        report = self.df.isna().sum().sort_values(ascending=False).to_frame("Missing")

        report["Percentage"] = (report["Missing"] / len(self.df) * 100).round(2)

        return report

    def constant_columns(self) -> list[str]:
        """
        Return columns with only one unique value.
        """

        return [column for column in self.df.columns if self.df[column].nunique() == 1]

    def cardinality(self) -> pd.Series:
        """
        Number of unique values per feature.
        """

        return self.df.nunique().sort_values(ascending=False)

    def numerical_summary(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for numerical columns.
        """

        return self.df.describe().T

    def memory_usage(self) -> float:
        """
        Calculate total memory usage in MB.
        """

        return round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)

    def save_reports(self) -> None:
        """
        Save all generated reports to CSV files.
        """

        METRICS_DIR.mkdir(exist_ok=True, parents=True)

        self.basic_information().to_csv(
            METRICS_DIR / "feature_information.csv", index=True
        )

        self.missing_report().to_csv(METRICS_DIR / "missing_report.csv", index=True)

        self.numerical_summary().to_csv(
            METRICS_DIR / "numerical_summary.csv", index=True
        )

        self.cardinality().to_csv(
            METRICS_DIR / "cardinality.csv", header=["Unique Values"]
        )

        logger.info("All quality reports saved successfully.")
