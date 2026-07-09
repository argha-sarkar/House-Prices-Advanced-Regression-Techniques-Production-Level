import pandas as pd

from src.preprocessing.transformers import HouseFeatureEngineer


def test_house_age_feature():
    df = pd.DataFrame(
        {
            "YrSold": [2010],
            "YearBuilt": [2000],
            "YearRemodAdd": [2005],
            "GarageYrBlt": [2000],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [0],
            "TotalBsmtSF": [1000],
            "1stFlrSF": [1000],
            "2ndFlrSF": [500],
            "OpenPorchSF": [50],
            "EnclosedPorch": [0],
            "3SsnPorch": [0],
            "ScreenPorch": [0],
            "TotRmsAbvGrd": [7],
            "BedroomAbvGr": [3],
            "OverallQual": [8],
            "GrLivArea": [1500],
            "BsmtFinSF1": [700],
        }
    )

    transformer = HouseFeatureEngineer()
    transformed = transformer.transform(df)

    assert transformed["HouseAge"].iloc[0] == 10
