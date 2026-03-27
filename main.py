import logging
import os
from pathlib import Path
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf
import mlflow
import mlflow.pyfunc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HotDog Classifier API",
    description="API for classifying images as hotdog or not hotdog",
    version="1.0.0"
)

# Configuration
IMAGE_SIZE = 224
MODEL_ARTIFACT_PATH = os.getenv(
    "MODEL_ARTIFACT_PATH",
    "mlruns/1/models/m-076ae1d921f341609a4a43991758072c/artifacts",
)
HOTDOG_CLASSES = ["not_hotdog", "hotdog"]

# Global model instance
model = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    confidence: float
    probabilities: dict
    success: bool


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool


def load_model():
    """Load the trained MLflow model"""
    global model
    try:
        logger.info(f"Loading model from: {MODEL_ARTIFACT_PATH}")
        model = mlflow.pyfunc.load_model(MODEL_ARTIFACT_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    # Keep preprocessing consistent with training data pipeline:
    # model includes a Rescaling layer, so inputs must remain in [0, 255].
    image_array = np.array(image, dtype=np.float32)
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up API server...")
    if load_model():
        logger.info("✓ Model loaded successfully")
    else:
        logger.warning("⚠ Model failed to load - predictions will fail")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image is a hotdog or not
    
    Parameters:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        Prediction with class and confidence score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess image
        image_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(image_array)
        
        # Handle output format (could be [batch_size, 2] for softmax)
        if predictions.ndim == 2:
            pred_probs = predictions[0]  # Get first (and only) sample
        else:
            pred_probs = predictions
        
        # Get class and confidence
        predicted_class_idx = np.argmax(pred_probs)
        confidence = float(np.max(pred_probs))
        predicted_class = HOTDOG_CLASSES[predicted_class_idx]
        
        # Build probabilities dict
        probabilities = {
            HOTDOG_CLASSES[i]: float(pred_probs[i]) 
            for i in range(len(HOTDOG_CLASSES))
        }
        
        logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.get("/info")
async def model_info():
    """Get information about the loaded model"""
    return JSONResponse({
        "model_path": MODEL_ARTIFACT_PATH,
        "image_size": IMAGE_SIZE,
        "classes": HOTDOG_CLASSES,
        "model_loaded": model is not None
    })


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch prediction endpoint for multiple images
    
    Parameters:
        files: List of image files
        
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            
            image_array = preprocess_image(image)
            predictions = model.predict(image_array)
            
            if predictions.ndim == 2:
                pred_probs = predictions[0]
            else:
                pred_probs = predictions
            
            predicted_class_idx = np.argmax(pred_probs)
            confidence = float(np.max(pred_probs))
            predicted_class = HOTDOG_CLASSES[predicted_class_idx]
            
            results.append({
                "filename": file.filename,
                "prediction": predicted_class,
                "confidence": confidence
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
