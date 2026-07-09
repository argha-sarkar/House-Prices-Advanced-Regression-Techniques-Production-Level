from pathlib import Path

import joblib

# Get the directory where model_loader.py is located
BASE_DIR = Path(__file__).resolve().parent.parent

# Update this line to point to 'best_model.pkl'
model_path = BASE_DIR / "models" / "best_model.pkl"

# Load the model
MODEL = joblib.load(model_path)
