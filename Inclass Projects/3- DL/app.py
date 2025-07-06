import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from typing import Dict, Any
import uvicorn
from datetime import datetime

# Fashion-MNIST class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# CNN Model Architecture (same as training)
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 10)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# Initialize FastAPI app
app = FastAPI(
    title="Fashion-MNIST CNN Classifier",
    description="Deep Learning model for fashion item classification",
    version="1.0.0"
)

# Global variables
model = None
device = None
transform = None

def load_model():
    """Load the trained CNN model"""
    global model, device, transform
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = FashionCNN()
    
    # Load model weights
    model_path = "models/fashion_cnn_model.pt"
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model file not found at {model_path}")
        model = None
    
    # Define image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess uploaded image for model inference"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Apply preprocessing
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """Predict fashion item from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all predictions
            all_predictions = {}
            for i, class_name in enumerate(CLASS_NAMES):
                all_predictions[class_name] = float(probabilities[0][i])
        
        # Prepare response
        response = {
            "prediction": CLASS_NAMES[predicted.item()],
            "confidence": float(confidence.item()),
            "class_id": int(predicted.item()),
            "all_predictions": all_predictions,
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fashion-MNIST CNN Classifier API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict_image",
            "docs": "/docs"
        },
        "model_info": {
            "classes": CLASS_NAMES,
            "input_format": "28x28 grayscale images",
            "supported_formats": ["JPEG", "PNG", "BMP", "TIFF"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 