from pathlib import Path
from typing import Optional

import pandas as pd

from src.exceptions import DataValidationError, MissingColumnError
from src.logger import logger


class DataLoader:
    """
    Responsible for loading and validating datasets.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data.
        """
        logger.info(f"Loading dataset: {self.file_path}")

        if not self.file_path.exists():
            logger.error("Dataset file does not exist.")
            raise FileNotFoundError(self.file_path)

        try:
            df = pd.read_csv(self.file_path)

            logger.info("Dataset loaded successfully.")

            return df

        except Exception as error:
            logger.exception(error)
            raise

    @staticmethod
    def validate_target(
        df: pd.DataFrame,
        target: str,
    ) -> None:
        """
        Check whether the target column exists.
        """
        if target not in df.columns:
            raise MissingColumnError(f"{target} column not found.")

    @staticmethod
    def validate_empty(df: pd.DataFrame) -> None:
        """
        Ensure dataset is not empty.
        """
        if df.empty:
            raise DataValidationError("Dataset is empty.")

    @staticmethod
    def dataset_summary(
        df: pd.DataFrame,
    ) -> None:
        logger.info(f"Rows: {df.shape[0]}")

        logger.info(f"Columns: {df.shape[1]}")

        logger.info(
            f"Memory Usage: " f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB"
        )
