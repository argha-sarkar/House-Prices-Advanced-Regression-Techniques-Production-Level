from pydantic import BaseModel


class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: int
    TotalBsmtSF: float
    YearBuilt: int
    YearRemodAdd: int
    GarageYrBlt: int
    FullBath: int
    HalfBath: int
    BsmtFullBath: int
    BsmtHalfBath: int
    OpenPorchSF: float
    EnclosedPorch: float
    ScreenPorch: float
    ThreeSsnPorch: float
