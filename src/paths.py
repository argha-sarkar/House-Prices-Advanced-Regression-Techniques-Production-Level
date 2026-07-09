from pathlib import Path

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Figures
FIGURES_DIR = REPORTS_DIR / "figures"

# Metrics
METRICS_DIR = REPORTS_DIR / "metrics"
