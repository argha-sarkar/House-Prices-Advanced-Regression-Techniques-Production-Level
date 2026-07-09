from fastapi import FastAPI

from app.health import health_router
from app.predictor import router

app = FastAPI(
    title="House Price Prediction API", version="1.0"  # <--- Added a comma here
)

app.include_router(router)

app.include_router(health_router)
