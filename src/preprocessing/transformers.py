from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Create new engineered features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # -------------------------

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        # -------------------------

        X["RemodelAge"] = X["YrSold"] - X["YearRemodAdd"]

        # -------------------------

        X["GarageAge"] = X["YrSold"] - X["GarageYrBlt"]

        # -------------------------

        X["TotalBathrooms"] = (
            X["FullBath"]
            + 0.5 * X["HalfBath"]
            + X["BsmtFullBath"]
            + 0.5 * X["BsmtHalfBath"]
        )

        # -------------------------

        X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

        # -------------------------

        X["TotalPorchSF"] = (
            X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        )

        # -------------------------

        X["TotalRooms"] = X["TotRmsAbvGrd"] + X["BedroomAbvGr"]

        # -------------------------

        X["QualityLivingArea"] = X["OverallQual"] * X["GrLivArea"]

        X["LuxuryHome"] = ((X["OverallQual"] >= 8) & (X["GrLivArea"] > 2500)).astype(
            int
        )

        X["FinishedBasementRatio"] = X["BsmtFinSF1"] / (X["TotalBsmtSF"] + 1)

        return X
