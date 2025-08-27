# EECE490-690: Machine Learning

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)

**Course**: EECE490-690 Machine Learning  
**Institution**: [University Name]  
**Academic Year**: 2024-2025  

## Table of Contents

- [Course Overview](#course-overview)
- [Repository Structure](#repository-structure)
- [Learning Objectives](#learning-objectives)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Course Modules](#course-modules)
- [In-Class Projects](#in-class-projects)
- [Assessment](#assessment)
- [Prerequisites](#prerequisites)
- [Resources](#resources)
- [Contributing](#contributing)
- [Contact](#contact)

## Course Overview

This repository contains comprehensive materials for EECE490-690 Machine Learning, a graduate-level course covering theoretical foundations and practical applications of machine learning algorithms. The course emphasizes hands-on experience with real-world datasets and modern ML frameworks, progressing from basic statistical methods to advanced deep learning and transformer architectures.

### Course Philosophy
- **Theory-Practice Integration**: Combining mathematical foundations with practical implementation
- **Industry-Relevant Skills**: Focus on tools and techniques used in production ML systems
- **Progressive Learning**: Building complexity from basic algorithms to state-of-the-art models
- **Real-World Applications**: Using authentic datasets and deployment scenarios

## Repository Structure

```
EECE490-690/
├── Chapters/                          # Core learning modules
│   ├── C1 - Machine Learning in the Real World/
│   ├── C2 - Preparing Data for ML Algorithms/
│   ├── C3 - Exploratory Data Analysis/
│   ├── C4 - Supervised ML Algorithms/
│   ├── C5 - Unsupervised ML Algorithms/
│   ├── C6 - Statistical Model Tuning and Testing/
│   ├── C7 - Deep Learning/
│   └── C8 - Transformers & LLMs/
├── Inclass Projects/                  # Hands-on practical projects
│   ├── 1- Data/                      # EDA and data preprocessing
│   ├── 2- ML/                        # Classical ML pipeline
│   └── 3- DL/                        # Deep learning deployment
└── README.md                         # This file
```

## Learning Objectives

Upon successful completion of this course, students will be able to:

### Technical Competencies
- **Data Processing**: Master techniques for data cleaning, feature engineering, and preprocessing
- **Algorithm Implementation**: Implement and optimize both supervised and unsupervised learning algorithms
- **Deep Learning**: Design, train, and deploy neural networks using PyTorch
- **Model Evaluation**: Apply rigorous statistical methods for model validation and comparison
- **Production Deployment**: Deploy ML models as scalable web services using modern DevOps practices

### Analytical Skills
- **Problem Formulation**: Translate business problems into appropriate ML formulations
- **Model Selection**: Choose optimal algorithms based on data characteristics and requirements
- **Performance Analysis**: Interpret and communicate model performance to technical and non-technical stakeholders
- **Ethical Considerations**: Understand and address bias, fairness, and interpretability in ML systems

### Professional Skills
- **MLOps Practices**: Implement version control, containerization, and deployment pipelines
- **Team Collaboration**: Work effectively on ML projects using industry-standard tools
- **Documentation**: Create clear, reproducible documentation for ML experiments and deployments

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Jupyter Notebooks**: Interactive development environment
- **PyTorch 1.13+**: Deep learning framework
- **scikit-learn**: Classical machine learning algorithms
- **pandas & NumPy**: Data manipulation and numerical computing

### Visualization & Analysis
- **matplotlib & seaborn**: Statistical visualization
- **plotly**: Interactive visualizations
- **BertViz**: Transformer attention visualization

### Production & Deployment
- **FastAPI**: Web service framework
- **Docker**: Containerization platform
- **Postman**: API testing
- **Hetzner Cloud**: Cloud deployment platform

### Specialized Libraries
- **Transformers (Hugging Face)**: Pre-trained language models
- **OpenCV**: Computer vision tasks
- **NLTK**: Natural language processing
- **joblib**: Model persistence

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Git version control
- Basic understanding of linear algebra and statistics
- Familiarity with Jupyter notebooks

### Installation

#### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd EECE490-690

# Create conda environment
conda env create -f environment.yml
conda activate eece490-ml

# Launch Jupyter
jupyter notebook
```

#### Option 2: pip Installation
```bash
# Clone the repository
git clone <repository-url>
cd EECE490-690

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Verification
Test your installation by running the first notebook in Chapter 2:
```bash
jupyter notebook "Chapters/C2 - Preparing Data for Statistical Machine Learning Algorithms/1- Encoding Categorical Features in Tabular Data.ipynb"
```

## Course Modules

### Chapter 1: Machine Learning in the Real World
**Duration**: 2 weeks  
**Focus**: Introduction to ML concepts, applications, and industry practices

### Chapter 2: Preparing Data for ML Algorithms
**Duration**: 2 weeks  
**Topics**:
- Encoding categorical features in tabular data
- Image data representation and preprocessing
- Text data vectorization and encoding
- Data quality assessment and cleaning

**Key Skills**: Feature engineering, data preprocessing pipelines, handling different data modalities

### Chapter 3: Exploratory Data Analysis
**Duration**: 2 weeks  
**Topics**:
- Statistical analysis of tabular data
- Image data exploration and visualization
- Time series pattern analysis
- Text data exploration and preprocessing

**Key Skills**: Statistical visualization, pattern recognition, data profiling

### Chapter 4: Supervised Machine Learning Algorithms
**Duration**: 3 weeks  
**Topics**:
- Classification algorithms and evaluation
- Regression modeling and metrics
- Linear regression implementation
- Performance evaluation frameworks

**Key Skills**: Algorithm selection, hyperparameter tuning, cross-validation

### Chapter 5: Unsupervised Machine Learning Algorithms
**Duration**: 2 weeks  
**Topics**:
- Density-based clustering (DBSCAN)
- K-means clustering implementation
- Principal Component Analysis (PCA)
- Dimensionality reduction techniques

**Key Skills**: Clustering evaluation, feature selection, anomaly detection

### Chapter 6: Statistical Model Tuning and Testing
**Duration**: 2 weeks  
**Topics**:
- Unsupervised learning evaluation
- Supervised learning validation strategies
- Statistical significance testing
- Model comparison frameworks

**Key Skills**: Rigorous evaluation methodology, statistical testing, model selection

### Chapter 7: Deep Learning
**Duration**: 4 weeks  
**Topics**:
- Neural network fundamentals in PyTorch
- Convolutional Neural Networks (CNNs)
- Advanced CNN architectures
- Image augmentation techniques
- Optimization and regularization
- Object detection with YOLO

**Key Skills**: Deep learning implementation, computer vision, model optimization

### Chapter 8: Transformers & Large Language Models
**Duration**: 3 weeks  
**Topics**:
- Multi-head attention visualization
- Vision Transformers for image classification
- Large language model exploration (Falcon-7B)
- Domain-specific applications (Arabic NLP)
- Fine-tuning with LoRA

**Key Skills**: Transformer architecture, LLM applications, transfer learning

## In-Class Projects

### Project 1: Exploratory Data Analysis
**Timeline**: Week 4 (90 minutes)  
**Dataset**: Student Performance (50 records)  
**Objective**: Transform raw CSV data into actionable insights through systematic EDA

**Learning Outcomes**:
- Master systematic EDA methodology
- Create meaningful visualizations
- Identify key correlations and patterns
- Generate actionable insights from data

**Deliverables**: Complete Jupyter notebook with visualizations and insights

### Project 2: ML Microservice
**Timeline**: Week 8 (90 minutes)  
**Dataset**: SMS Spam Classification (5,000+ messages)  
**Objective**: Build, deploy, and containerize an ML classification service

**Learning Outcomes**:
- Implement end-to-end ML pipeline
- Create production-ready API endpoints
- Apply containerization best practices
- Test deployed services

**Deliverables**: Functional dockerized API with documented endpoints

### Project 3: Deep Learning Deployment
**Timeline**: Week 14 (90 minutes)  
**Dataset**: Fashion-MNIST  
**Objective**: Train CNN locally and deploy to cloud infrastructure

**Learning Outcomes**:
- Implement CNN training pipelines
- Create production ML APIs
- Master container orchestration
- Deploy to cloud infrastructure


### Online Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Development Tools
- [Anaconda Distribution](https://www.anaconda.com/products/distribution)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Postman API Platform](https://www.postman.com/)
- [Git SCM](https://git-scm.com/)

### Datasets
- All datasets are provided within the repository
- External datasets are accessed through standard ML libraries
- Cloud datasets are referenced with appropriate licenses


## Contact
Introduction to Machine Learning (FA25)
EECE490/EECE690

-----------------------------

Ammar Mohanna

Email: am288@aub.edu.lb
Office hours: RGB 405 - Tuesday 11:00-13:00, Thursday 13:30-15:30 (first come, first served)

-----------------------------

TAs:
Mohammad Zbeeb - mbz02@mail.aub.edu



---

**Last Updated**: August 2024  
**Version**: 1.0  
**License**: Academic Use Only

This repository is maintained as part of the EECE490-690 Machine Learning course curriculum. All materials are designed for educational purposes and reflect current industry best practices in machine learning and data science.
