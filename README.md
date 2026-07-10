🏠 House Price Prediction - Advanced Regression Techniques

Production-Grade Machine Learning Project built using Python, Scikit-Learn, XGBoost, LightGBM, CatBoost, FastAPI, Docker, and modern MLOps practices.

📌 Project Overview

Buying or selling a house is one of the biggest financial decisions people make. Accurately estimating a property's value is important for homeowners, buyers, real estate agents, banks, and insurance companies.

This project predicts the selling price of residential homes using the House Prices: Advanced Regression Techniques Kaggle dataset. Rather than creating a simple machine learning notebook, the goal was to design a production-ready machine learning system following software engineering best practices.

The project covers the complete machine learning lifecycle, including:

Data analysis
Data validation
Exploratory Data Analysis (EDA)
Feature engineering
Data preprocessing
Model training
Hyperparameter optimization
Model explainability
Error analysis
REST API deployment
Docker containerization
Unit testing
Continuous Integration (CI)
🎯 Objectives

The primary objectives of this project are:

Predict house prices with high accuracy.
Build a reusable machine learning pipeline.
Prevent data leakage using Scikit-Learn Pipelines.
Compare multiple regression algorithms.
Optimize model performance using Optuna.
Explain model predictions using SHAP.
Deploy the model as a REST API using FastAPI.
Follow production-grade software engineering practices.
📊 Dataset

Competition

House Prices: Advanced Regression Techniques (Kaggle)

The dataset contains detailed information about residential homes, including:

Lot size
Number of rooms
Basement size
Garage information
Year built
Neighborhood
Exterior quality
Interior quality
Sale price
Target Variable
SalePrice

The goal is to predict the final selling price of each house.


```

🏗 Project Structure
house-price-prediction/
│
├── app/
│   ├── main.py
│   ├── predictor.py
│   ├── schemas.py
│   ├── model_loader.py
│   └── health.py
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── src/
│   ├── preprocessing/
│   ├── visualization/
│   ├── explainability/
│   ├── optimization/
│   ├── models/
│   ├── logger.py
│   ├── config.py
│   ├── paths.py
│   └── data_loader.py
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

```

⚙ Technologies Used
Programming Language
Python 3.12
Data Analysis
Pandas
NumPy
SciPy
Visualization
Matplotlib
Seaborn
Plotly
Machine Learning
Scikit-Learn
XGBoost
LightGBM
CatBoost
Hyperparameter Optimization
Optuna
Explainable AI
SHAP
API Development
FastAPI
Pydantic
Deployment
Docker
Uvicorn
Utilities
Joblib
Loguru
PyYAML
Testing
Pytest
🔄 Machine Learning Workflow

The project follows a structured workflow:

Raw Dataset
        ↓
Data Validation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Hyperparameter Optimization
        ↓
Model Evaluation
        ↓
Explainability
        ↓
REST API
        ↓
Docker Deployment
📈 Exploratory Data Analysis

The EDA phase includes:

Missing value analysis
Duplicate detection
Feature distributions
Correlation heatmap
Outlier analysis
Numerical summary
Categorical feature analysis
Target distribution
Scatter plots
Boxplots
Violin plots

Key observations:

SalePrice is right-skewed.
OverallQual has a strong positive relationship with SalePrice.
GrLivArea is highly correlated with house price.
Some numerical features contain extreme outliers.
Several categorical variables contain a large number of missing values.
🛠 Feature Engineering

Custom engineered features include:

HouseAge
RemodelAge
GarageAge
TotalBathrooms
TotalSF
TotalPorchSF
TotalRooms
QualityLivingArea
LuxuryHome
FinishedBasementRatio

Additional preprocessing includes:

Log transformation of SalePrice
Rare category grouping
Ordinal encoding
Missing value imputation
🤖 Models Evaluated

Multiple regression models were trained and compared.

Linear Regression
Ridge Regression
Lasso Regression
ElasticNet
Random Forest
Extra Trees
Gradient Boosting
HistGradientBoosting
XGBoost
LightGBM
CatBoost

The final model is selected based on cross-validation performance and evaluation metrics.

📏 Evaluation Metrics

The following metrics are used:

RMSE
MAE
R² Score

Cross-validation is used to obtain a more reliable estimate of model performance.

🎯 Hyperparameter Optimization

Optuna is used instead of GridSearchCV because it explores the search space more efficiently.

The optimization process includes:

Bayesian optimization
K-Fold Cross Validation
Automatic parameter search
Optimization history
Parameter importance analysis
🔍 Model Explainability

To better understand model predictions, the project includes:

SHAP Summary Plot
Feature Importance
Permutation Importance
Partial Dependence Plot
Residual Analysis
Error Analysis

These techniques help explain which features influence predictions and identify cases where the model performs poorly.

🌐 REST API

The trained pipeline is deployed using FastAPI.

Endpoints
GET /health

Returns the API health status.

POST /predict

Predicts the selling price of a house.

The API automatically validates incoming requests using Pydantic models.

🐳 Docker

The project is fully containerized using Docker.

Build the image:

docker build -t house-price-api .

Run the container:

docker run -p 8000:8000 house-price-api

Swagger documentation is available at:

http://localhost:8000/docs
🧪 Testing

Unit tests are written using Pytest.

The project includes tests for:

Data loading
Feature engineering
Preprocessing pipeline
Model training
REST API

Run all tests:

pytest
🚀 Future Improvements

Potential improvements include:

Stacking and blending ensembles
MLflow experiment tracking
DVC for dataset versioning
Model registry
Drift detection
Prometheus monitoring
Grafana dashboards
Cloud deployment on Azure, AWS, and Google Cloud
Automated retraining pipeline
📚 What I Learned

This project helped me strengthen my understanding of:

Production-grade machine learning pipelines
Feature engineering for tabular data
Hyperparameter optimization
Explainable AI
Software engineering for ML
REST API development
Docker containerization
Testing and continuous integration
Building reusable and maintainable machine learning systems
💡 Key Takeaways

This project is much more than a Kaggle submission. It demonstrates how to take a machine learning model from raw data to a deployable application using modern engineering practices.

The focus was not only on achieving good predictive performance but also on writing clean, maintainable, and production-ready code.

👤 Author

Argha Sarkar

If you found this project useful or have suggestions for improvement, feel free to connect or open an issue. Constructive feedback is always welcome.

⭐ Acknowledgements
Kaggle for providing the House Prices dataset and competition.
The open-source Python community for developing the libraries used throughout this project.
The Scikit-Learn, XGBoost, LightGBM, CatBoost, FastAPI, and Optuna teams for their excellent tools and documentation.