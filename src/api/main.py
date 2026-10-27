"""REST API for real-time sentiment predictions.

Run it with:
    uvicorn src.api.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive documentation (Swagger UI).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    TweetRequest,
)
from src.config import MODEL_PATH
from src.predict import load_model, predict_batch, predict_detailed


def get_model_path() -> Path:
    """Dependency that tells the endpoints where the model file is (easy to replace in tests)."""
    return MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, so the first request is not slow.
    # If the model has not been trained yet the API still starts: /health reports it
    # and the prediction endpoints answer with 503 until the model exists.
    try:
        load_model(get_model_path())
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Classifies the sentiment of tweets as Positive or Negative using a model trained on Sentiment140.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Twitter Sentiment Analysis API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health(model_path: Path = Depends(get_model_path)):
    """Check that the API is running and whether the trained model is available."""
    return {"status": "ok", "model_loaded": model_path.exists()}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: TweetRequest, model_path: Path = Depends(get_model_path)):
    """Predict the sentiment of a single tweet."""
    try:
        return predict_detailed(request.text, model_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_many(request: BatchRequest, model_path: Path = Depends(get_model_path)):
    """Predict the sentiment of several tweets in one request."""
    try:
        return {"predictions": predict_batch(request.texts, model_path)}
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
