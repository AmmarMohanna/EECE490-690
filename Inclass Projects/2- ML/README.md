# Project 2: Machine Learning - SMS Spam Classification

## Overview
This project provides hands-on experience with both supervised and unsupervised machine learning techniques using an SMS spam classification dataset. Students will learn to build, evaluate, and deploy a complete ML pipeline from data preprocessing to model deployment.

## Files
- `Project2_ML_SMS_Spam_Classification.ipynb` - Main Jupyter notebook with complete ML workflow
- `Project2_ML_SMS_Spam_Classification_FILLIN.ipynb` - Fill-in version for classroom instruction
- `sms_spam_dataset.csv` - SMS spam dataset with 5,000+ messages
- `requirements.txt` - Python package dependencies
- `environment.yml` - Conda environment file

## Dataset Description
The `sms_spam_dataset.csv` contains SMS messages with the following features:

### Features:
- **text**: The SMS message content
- **label**: Classification (spam/ham)
- **length**: Message length in characters
- **word_count**: Number of words in message
- **exclamation_count**: Number of exclamation marks
- **question_count**: Number of question marks
- **currency_symbols**: Count of currency symbols (£, $, €)
- **numbers_count**: Count of numeric digits

## Learning Objectives

### Supervised Learning:
- Text preprocessing and feature engineering
- TF-IDF vectorization
- Logistic Regression classification
- Naive Bayes classification
- Random Forest classification
- Model evaluation and comparison
- Hyperparameter tuning with GridSearchCV

### Unsupervised Learning:
- K-means clustering on text features
- PCA for dimensionality reduction
- t-SNE visualization
- Anomaly detection for spam identification
- Topic modeling with LDA

### Model Deployment:
- Model persistence with joblib
- FastAPI web service creation
- Docker containerization
- API testing with Postman

## Setup Instructions

### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate ml-lab

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
- **For complete version**: Open `Project2_ML_SMS_Spam_Classification.ipynb`
- **For fill-in version**: Open `Project2_ML_SMS_Spam_Classification_FILLIN.ipynb`

## Project Structure (90 minutes)

### Phase 1: Data Exploration and Preprocessing (0-20 min)
- Load and explore the SMS dataset
- Text preprocessing (cleaning, tokenization)
- Feature engineering from text data
- Visualize spam vs ham distributions

### Phase 2: Supervised Learning (20-50 min)
- Train multiple classification models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Evaluate models using accuracy, precision, recall, F1-score
- Compare model performance
- Hyperparameter tuning

### Phase 3: Unsupervised Learning (50-70 min)
- K-means clustering on TF-IDF features
- PCA for dimensionality reduction
- Anomaly detection for spam identification
- Visualize clusters and patterns

### Phase 4: Model Deployment (70-90 min)
- Save best performing model
- Create FastAPI web service
- Test API with sample requests
- Optional: Docker containerization

## Technical Skills Covered:
- Text preprocessing and NLP techniques
- Feature extraction (TF-IDF, Count Vectorizer)
- Supervised learning algorithms
- Unsupervised learning techniques
- Model evaluation metrics
- Cross-validation and hyperparameter tuning
- Model persistence and deployment
- API development with FastAPI

## Expected Outcomes
By the end of this session, students will be able to:
1. Build end-to-end ML pipelines for text classification
2. Apply both supervised and unsupervised learning techniques
3. Evaluate and compare different ML models
4. Deploy ML models as web services
5. Understand the complete ML workflow from data to deployment

## Tips for Success
- Follow the notebook structure systematically
- Pay attention to text preprocessing steps
- Compare different models objectively
- Document your findings and model performance
- Test your API endpoints thoroughly
- Be prepared to discuss your model choices

## Next Steps
After completing this project, students will be ready for:
- Advanced NLP techniques
- Deep learning for text classification
- Production ML deployment
- MLOps practices
- Advanced model optimization 