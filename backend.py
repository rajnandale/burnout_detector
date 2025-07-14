from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from improved_burnout_detector import ImprovedBurnoutDetector
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="Improved Burnout Detection API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = None

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    risk_level: str
    confidence: float
    probabilities: dict
    explanation: list
    recommendations: list

def get_recommendations(risk_level: str) -> list:
    """Get recommendations based on risk level"""
    recommendations = {
        "Low Risk": [
            "Continue maintaining your current work-life balance",
            "Keep up with your positive workplace practices",
            "Consider mentoring others who might be struggling"
        ],
        "Moderate Risk": [
            "Take regular breaks throughout your workday",
            "Practice stress management techniques",
            "Consider discussing workload with your manager",
            "Prioritize self-care activities outside work"
        ],
        "High Risk": [
            "Consider speaking with HR or a mental health professional",
            "Take time off if possible to recharge",
            "Discuss workload reduction with your manager",
            "Seek support from employee assistance programs",
            "Consider professional counseling or therapy"
        ]
    }
    return recommendations.get(risk_level, ["Please consult with HR for personalized guidance"])

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global detector
    
    # Check if trained model exists
    model_path = "improved_burnout_detector.pkl"
    
    if os.path.exists(model_path):
        try:
            detector = ImprovedBurnoutDetector.load_model(model_path)
            print("✅ Loaded pre-trained model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Training new model...")
            detector = ImprovedBurnoutDetector()
            detector.train()
            detector.save_model(model_path)
    else:
        print("🔄 Training new model...")
        detector = ImprovedBurnoutDetector()
        detector.train()
        detector.save_model(model_path)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Improved Burnout Detection API",
        "version": "2.0.0",
        "status": "healthy",
        "model_loaded": detector is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_burnout(data: TextInput):
    """Predict burnout level from text input"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Make prediction
        result = detector.predict(data.text)
        
        # Add recommendations
        recommendations = get_recommendations(result['risk_level'])
        
        return PredictionResponse(
            risk_level=result['risk_level'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            explanation=result['explanation'],
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Classifier",
        "features": [
            "Negative sentiment score",
            "Positive sentiment score", 
            "Compound sentiment score",
            "Word count",
            "Exhaustion keyword count",
            "Stress keyword count",
            "Negative emotions count",
            "Workload keyword count",
            "Burnout-specific keyword count"
        ],
        "risk_levels": ["Low Risk", "Moderate Risk", "High Risk"],
        "training_data_size": "1000 synthetic samples (balanced)",
        "accuracy": "~85-90% on synthetic test data"
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain the model with new data"""
    global detector
    
    try:
        detector = ImprovedBurnoutDetector()
        accuracy = detector.train()
        detector.save_model("improved_burnout_detector.pkl")
        
        return {
            "message": "Model retrained successfully",
            "accuracy": f"{accuracy:.2%}",
            "model_saved": True
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 