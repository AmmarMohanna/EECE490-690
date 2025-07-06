# Project 3: Deep Learning - CNN Training and Deployment

## Overview
This project provides hands-on experience with training a Convolutional Neural Network (CNN) locally and deploying it to a live server endpoint. Students will learn the complete deep learning pipeline from model training to production deployment using Docker and cloud infrastructure.

## Files
- `Project3_DL_CNN_Training_Deployment.ipynb` - Main Jupyter notebook with complete DL workflow
- `FILLIN_Project3_DL_CNN_Training_Deployment.ipynb` - Fill-in version for classroom instruction
- `app.py` - FastAPI application for model serving
- `Dockerfile` - Multi-stage Docker build configuration
- `docker-compose.yml` - Container orchestration setup
- `requirements.txt` - Python package dependencies
- `environment.yml` - Conda environment file
- `sample_images/` - Test images for inference
- `models/` - Directory for saved model files

## Learning Objectives

### Deep Learning:
- CNN architecture design and implementation
- Training loops with PyTorch
- Model checkpointing and serialization
- Image preprocessing and data augmentation
- Transfer learning concepts

### MLOps & Deployment:
- Model containerization with Docker
- FastAPI web service creation
- Multi-stage Docker builds
- Cloud deployment on Hetzner
- Production inference pipelines
- API testing and validation

### DevOps Skills:
- Container orchestration
- SSH and remote deployment
- Environment management
- Service monitoring and health checks

## Dataset Description
The project uses **Fashion-MNIST** dataset for efficient training:

### Features:
- **Images**: 28x28 grayscale fashion items
- **Classes**: 10 categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Format**: Normalized pixel values (0-1)

## Setup Instructions

### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate dl-deploy

# Launch Jupyter Notebook
jupyter notebook
```

### Option 2: Using pip
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### 3. Open the Project Notebook
- **For complete version**: Open `Project3_DL_CNN_Training_Deployment.ipynb`
- **For fill-in version**: Open `Project3_DL_CNN_Training_Deployment_FILLIN.ipynb`

## Project Structure (90 minutes)

### Phase 1: Kick-off and Setup (0-5 min)
- Project overview and objectives
- Environment verification
- Dataset loading and exploration

### Phase 2: Local CNN Training (5-35 min)
- CNN architecture design (6-layer network)
- Training loop implementation
- Model evaluation and validation
- Model serialization (model.pt export)

### Phase 3: API Development (35-55 min)
- FastAPI application structure
- `/predict_image` endpoint implementation
- `/health` endpoint for monitoring
- Local API testing with sample images

### Phase 4: Containerization (55-75 min)
- Multi-stage Dockerfile creation
- Container build and testing
- Local container inference validation
- Image optimization strategies

### Phase 5: Cloud Deployment (75-90 min)
- Hetzner VM preparation
- Container registry setup
- Remote deployment with docker-compose
- Production API testing

## Technical Skills Covered:
- PyTorch CNN implementation
- Image classification pipelines
- Model persistence and loading
- RESTful API development
- Docker containerization
- Cloud infrastructure deployment
- Production monitoring

## Architecture Overview

```
Local Development → Docker Container → Cloud Deployment
     ↓                    ↓                  ↓
[CNN Training]     [API Service]      [Hetzner VM]
[Model Export]     [Health Check]     [Public URL]
[Local Testing]    [Inference]        [Monitoring]
```

## API Endpoints

### Health Check
```http
GET /health
Response: {"status": "healthy", "model_loaded": true}
```

### Image Prediction
```http
POST /predict_image
Content-Type: multipart/form-data
Body: image file

Response: {
  "prediction": "T-shirt/top",
  "confidence": 0.95,
  "class_id": 0,
  "all_predictions": {...}
}
```

## Prerequisites
- Python 3.8+
- PyTorch 1.8+
- Docker installed
- SSH access to deployment server
- Basic understanding of CNNs

## Expected Outcomes
By the end of this session, students will be able to:
1. Design and train CNN models with PyTorch
2. Create production-ready ML APIs with FastAPI
3. Containerize ML applications with Docker
4. Deploy models to cloud infrastructure
5. Monitor and test production ML services
6. Understand the complete MLOps workflow

## Tips for Success
- Test each phase thoroughly before moving to the next
- Pay attention to image preprocessing consistency
- Verify container builds locally before deployment
- Monitor resource usage during training
- Test API endpoints with various image types
- Document any deployment issues for troubleshooting

## Troubleshooting

### Common Issues:
- **CUDA out of memory**: Reduce batch size or use CPU training
- **Container build fails**: Check Docker daemon and image layers
- **API connection refused**: Verify port mapping and firewall settings
- **Model loading errors**: Ensure consistent PyTorch versions
- **SSH connection issues**: Check key permissions and server status

### Performance Optimization:
- Use GPU acceleration when available
- Implement efficient image preprocessing
- Optimize Docker image layers
- Configure appropriate batch sizes
- Monitor memory usage during inference

## Next Steps
After completing this project, students will be ready for:
- Advanced CNN architectures (ResNet, EfficientNet)
- Production MLOps pipelines
- Kubernetes orchestration
- Model versioning and A/B testing
- Continuous integration/deployment
- Advanced monitoring and logging 