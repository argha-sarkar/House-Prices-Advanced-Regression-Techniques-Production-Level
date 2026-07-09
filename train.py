import os

import joblib

from src.config import settings
from src.data_loader import DataLoader
from src.data_quality import DataQualityAnalyzer
from src.explainability.error_analysis import largest_errors
from src.explainability.feature_importance import plot_feature_importance
from src.explainability.residuals import residual_plot
from src.explainability.shap_analysis import shap_summary
from src.models.train import train_models
from src.optimization.tuner import tune_model
from src.paths import RAW_DATA_DIR
from src.preprocessing.encoder import encode_quality
from src.preprocessing.feature_engineering import group_rare_categories
from src.preprocessing.pipeline import build_preprocessor
from src.preprocessing.splitter import split_data
from src.visualization.bivariate import BivariateAnalysis
from src.visualization.multivariate import MultivariateAnalysis
from src.visualization.univariate import UnivariateAnalysis

# 1. Data Loading and Quality
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
loader = DataLoader(TRAIN_PATH)
train_df = loader.load_data()
loader.validate_empty(train_df)
loader.validate_target(train_df, settings["target_column"])

quality = DataQualityAnalyzer(train_df)
quality.save_reports()

# 2. Visualization
uni = UnivariateAnalysis(train_df)
uni.histogram("SalePrice")
bi = BivariateAnalysis(train_df)
bi.scatter("OverallQual", "SalePrice")
multi = MultivariateAnalysis(train_df)
multi.correlation_heatmap()

# 3. Preprocessing
train_df = group_rare_categories(train_df, column="Neighborhood", threshold=10)
train_df = encode_quality(train_df, column="KitchenQual")

X = train_df.drop(columns="SalePrice")
y = train_df["SalePrice"]

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

preprocessor = build_preprocessor(train_df)
X_train_processed = preprocessor.fit_transform(X_train)

# Make sure the models directory exists
import os

# --- ADD THESE LINES HERE ---
import joblib

os.makedirs("models", exist_ok=True)
# Save the preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")


X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# 4. Training
# Ensure train_models returns: results_df, model_object
results, best_model = train_models(X_train_processed, X_test_processed, y_train, y_test)
print(results)

# 5. Optimization
study = tune_model(X, y)
print(f"Best Params: {study.best_params}")
print(f"Best Value: {study.best_value}")

# 6. Explainability
plot_feature_importance(best_model, feature_names)
shap_summary(best_model, X_train_processed)
residual_plot(best_model, X_test_processed, y_test)

errors = largest_errors(best_model, X_test_processed, y_test)
print(errors.head(10))
