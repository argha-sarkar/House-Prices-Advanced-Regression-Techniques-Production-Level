from pathlib import Path

import joblib

from src.paths import MODELS_DIR

MODELS_DIR.mkdir(
    exist_ok=True,
    parents=True,
)


def save_model(
    model,
    filename,
):

    path = MODELS_DIR / filename

    joblib.dump(
        model,
        path,
    )
