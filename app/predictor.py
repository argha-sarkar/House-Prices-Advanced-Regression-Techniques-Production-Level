# app/predictor.py
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

from app.model_loader import MODEL  # This loads your best_model.pkl
from app.schemas import HouseInput

# Load the preprocessor that was saved during training
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSOR = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")

router = APIRouter()


@router.post("/predict")
def predict(data: HouseInput):
    # Convert input data to DataFrame
    df = pd.DataFrame([data.model_dump()])

    # The preprocessor expects ALL original training columns.
    # Get the column names from the preprocessor's transformers.
    all_columns = []
    for name, transformer, columns in PREPROCESSOR.transformers_:
        all_columns.extend(columns)

    # Reindex to include all expected columns, filling missing ones with NaN.
    # The preprocessor's SimpleImputer steps will handle the NaN values.
    df = df.reindex(columns=all_columns)

    # Transform input using the same preprocessor used during training
    df_processed = PREPROCESSOR.transform(df)

    # Predict on the preprocessed features
    prediction = MODEL.predict(df_processed)

    return {"prediction": float(prediction[0])}
