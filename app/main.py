"""
FastAPI application for weather prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
import logging
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from app.prediction import get_predictor

# ==================== Logging ====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== FastAPI App ====================

app = FastAPI(
    title="Weather Prediction API",
    description="ML-powered weather prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== Middleware ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Static Files ====================

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== Pydantic Models ====================

class WeatherInput(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    season: int = Field(..., ge=1, le=4)
    Humidity: float = Field(..., ge=0, le=1)
    Wind_Speed_kmh: float = Field(..., alias="Wind Speed (km/h)")
    Wind_Bearing_degrees: float = Field(..., alias="Wind Bearing (degrees)", ge=0, le=360)
    Visibility_km: float = Field(..., alias="Visibility (km)")
    Pressure_millibars: float = Field(..., alias="Pressure (millibars)")
    Apparent_Temperature_C: float = Field(..., alias="Apparent Temperature (C)")
    Summary: str

    class Config:
        populate_by_name = True


class WeatherPrediction(BaseModel):
    temperature_celsius: float
    precipitation_type: str
    precipitation_probabilities: Dict[str, float]
    confidence: float


class BatchWeatherInput(BaseModel):
    inputs: List[WeatherInput]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    valid_precipitation_classes: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    temperature_column: str
    precipitation_column: str
    model_types: Dict[str, str]

# ==================== Startup / Shutdown ====================

@app.on_event("startup")
async def startup_event():
    try:
        predictor = get_predictor()
        logger.info("✅ Models loaded")
        logger.info(f"Classes: {predictor.artifacts['valid_classes']}")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔴 Shutting down API")

# ==================== API ROUTES ====================

@app.get("/", tags=["Root"])
async def root():
    """
    API root — JSON only (used by tests)
    """
    return {
        "message": "Welcome to Weather Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model-info",
            "ui": "/ui"
        }
    }


@app.get("/ui", response_class=HTMLResponse, tags=["UI"])
async def ui():
    """
    Frontend UI
    """
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Frontend UI not found</h1>"


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=predictor.artifacts is not None
        )
    except Exception:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=False
        )


@app.post("/predict", response_model=WeatherPrediction, tags=["Prediction"])
async def predict_weather(input_data: WeatherInput):
    try:
        predictor = get_predictor()
        input_dict = input_data.dict(by_alias=True)
        result = predictor.predict(input_dict)
        return WeatherPrediction(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_weather_batch(batch_input: BatchWeatherInput):
    try:
        predictor = get_predictor()
        inputs = [item.dict(by_alias=True) for item in batch_input.inputs]
        results = predictor.predict_batch(inputs)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    try:
        predictor = get_predictor()
        return ModelInfoResponse(**predictor.get_model_info())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ==================== Error Handlers ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/", "/health", "/predict",
                "/predict/batch", "/model-info", "/ui"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )

# ==================== Run ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
