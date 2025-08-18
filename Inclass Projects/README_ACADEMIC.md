# EECE490-690: Machine Learning and Deep Learning In-Class Projects

## Course Information
**Course Title:** EECE490-690 - Advanced Machine Learning and Deep Learning  
**Institution:** University Engineering Program  
**Academic Level:** Senior Undergraduate / Graduate (490/690)  
**Project Type:** Hands-on Laboratory Exercises  

## Course Overview
This repository contains three comprehensive in-class projects designed to provide students with practical, hands-on experience in the complete machine learning and deep learning pipeline. The projects progress systematically from foundational data analysis to advanced deep learning deployment, reflecting industry-standard practices and modern MLOps workflows.

## Learning Progression and Pedagogical Approach
The course follows a structured progression aligned with the fundamental blocks of machine learning education:

1. **Data Foundation** → Exploratory Data Analysis and Statistical Understanding
2. **Machine Learning** → Supervised and Unsupervised Learning Techniques  
3. **Deep Learning** → Neural Networks and Production Deployment

Each 90-minute session combines theoretical understanding with practical implementation, emphasizing real-world applicability and industry best practices.

---

## Project 1: Exploratory Data Analysis - Student Performance Dataset
**Academic Focus:** Statistical Analysis and Data Visualization  
**Duration:** 90 minutes  
**Prerequisites:** Basic statistics, Python programming  

### Educational Objectives
- **Statistical Literacy:** Understanding descriptive statistics, correlation analysis, and data distributions
- **Visualization Proficiency:** Creating meaningful plots using matplotlib and seaborn
- **Data Quality Assessment:** Identifying missing values, outliers, and data integrity issues
- **Feature Engineering:** Creating derived variables to enhance analytical insights
- **Critical Thinking:** Formulating hypotheses and validating them through data exploration

### Technical Implementation
- **Dataset:** 50 student records with academic and demographic features
- **Tools:** Jupyter Notebooks, pandas, numpy, matplotlib, seaborn
- **Key Techniques:** Correlation matrices, pair plots, distribution analysis, categorical encoding
- **Deliverables:** Visual insights report, correlation analysis, pattern identification

### Academic Relevance
This project establishes the foundation for all subsequent machine learning work by teaching students to understand their data before modeling. It emphasizes the critical importance of exploratory data analysis in the scientific method and evidence-based decision making.

---

## Project 2: Machine Learning - SMS Spam Classification
**Academic Focus:** Supervised and Unsupervised Learning Applications  
**Duration:** 90 minutes  
**Prerequisites:** Project 1 completion, basic ML theory  

### Educational Objectives
- **Text Processing:** Natural Language Processing (NLP) fundamentals and feature extraction
- **Classification Methods:** Comparative analysis of multiple ML algorithms
- **Model Evaluation:** Understanding precision, recall, F1-score, and cross-validation
- **Unsupervised Learning:** Clustering and dimensionality reduction techniques
- **API Development:** Translating ML models into production-ready services

### Technical Implementation
- **Dataset:** 5,000+ SMS messages with engineered features
- **Supervised Learning:** Logistic Regression, Naive Bayes, Random Forest
- **Unsupervised Learning:** K-means clustering, PCA, t-SNE visualization, LDA topic modeling
- **Feature Engineering:** TF-IDF vectorization, count features, text preprocessing
- **Deployment:** FastAPI web service, model persistence with joblib
- **Containerization:** Docker implementation for reproducible deployments

### Academic Relevance
This project bridges theoretical machine learning concepts with practical implementation, teaching students the complete ML pipeline from data preprocessing to model deployment. It emphasizes comparative analysis, model selection, and the transition from research to production environments.

---

## Project 3: Deep Learning - CNN Training and Deployment
**Academic Focus:** Neural Networks and Production MLOps  
**Duration:** 90 minutes  
**Prerequisites:** Projects 1-2 completion, basic neural network theory  

### Educational Objectives
- **Deep Learning Architecture:** Understanding Convolutional Neural Networks (CNNs)
- **Training Procedures:** Backpropagation, gradient descent, model optimization
- **Production Deployment:** End-to-end MLOps pipeline implementation
- **Cloud Computing:** Infrastructure management and scalable deployment
- **Performance Monitoring:** Health checks, logging, and production monitoring

### Technical Implementation
- **Model Architecture:** 6-layer CNN with batch normalization and dropout regularization
- **Dataset:** Fashion-MNIST (28x28 grayscale images, 10 classes)
- **Framework:** PyTorch for model development and training
- **API Service:** FastAPI with image preprocessing and inference pipelines
- **Containerization:** Multi-stage Docker builds for production optimization
- **Cloud Deployment:** Hetzner VM deployment with docker-compose orchestration
- **Monitoring:** Health check endpoints and comprehensive logging

### Advanced Features
- **Model Serialization:** PyTorch state dict management and version control
- **Image Processing Pipeline:** Robust preprocessing with PIL/Pillow
- **Error Handling:** Production-grade exception management and logging
- **Security:** Non-root Docker containers and secure deployment practices

### Academic Relevance
This capstone project integrates all previous learning into a complete production system, teaching students the complexities of deploying ML models in real-world environments. It emphasizes the intersection of machine learning, software engineering, and cloud computing that defines modern AI engineering roles.

---

## Technical Infrastructure

### Development Environment
- **Python Version:** 3.9+
- **Package Management:** Conda and pip support with environment files
- **Containerization:** Docker with multi-stage builds and docker-compose
- **Cloud Platform:** Hetzner VM infrastructure
- **Version Control:** Git-based project management

### Dependencies and Tools
- **Data Science Stack:** pandas, numpy, scipy, matplotlib, seaborn
- **Machine Learning:** scikit-learn, PyTorch, torchvision
- **Web Development:** FastAPI, uvicorn, Pillow
- **Development Tools:** Jupyter notebooks, Docker, SSH
- **Testing Framework:** Custom validation and health check systems

---

## Assessment and Learning Outcomes

### Knowledge Acquisition
Upon completion of all three projects, students will demonstrate:
1. **Statistical Competency:** Ability to perform comprehensive exploratory data analysis
2. **ML Proficiency:** Understanding of supervised and unsupervised learning paradigms
3. **Deep Learning Expertise:** Practical experience with neural network architectures
4. **Engineering Skills:** Production deployment and MLOps best practices
5. **Critical Thinking:** Ability to select appropriate techniques for different problem domains

### Professional Preparedness
The project sequence prepares students for:
- **Industry Roles:** Machine Learning Engineer, Data Scientist, AI Researcher positions
- **Advanced Coursework:** Graduate-level AI/ML research and specialized applications
- **Technical Leadership:** Understanding of complete ML product development lifecycle
- **Innovation Capacity:** Ability to prototype and deploy novel ML solutions

---

## Course Structure and Methodology

### Session Format (90 minutes each)
1. **Conceptual Introduction** (10-15 minutes): Theoretical foundations and context
2. **Live Demonstration** (20-30 minutes): Instructor-led implementation walkthrough
3. **Guided Practice** (40-50 minutes): Student hands-on development with instructor support
4. **Presentation and Discussion** (10-15 minutes): Student presentations and peer learning

### Pedagogical Principles
- **Active Learning:** Immediate application of theoretical concepts
- **Scaffolded Instruction:** Progressive complexity building on previous knowledge
- **Collaborative Learning:** Peer interaction and knowledge sharing
- **Authentic Assessment:** Real-world problem solving and implementation
- **Reflective Practice:** Documentation and presentation of learning outcomes

---

## Repository Structure
```
/Inclass Projects/
├── README.md                    # Course overview and project summaries
├── 1- Data/                     # Project 1: Exploratory Data Analysis
│   ├── *.ipynb                  # Jupyter notebooks (complete and fill-in versions)
│   ├── student_performance.csv  # Academic performance dataset
│   └── environment.yml          # Conda environment specification
├── 2- ML/                       # Project 2: Machine Learning Pipeline
│   ├── *.ipynb                  # ML classification notebooks
│   ├── sms_spam_dataset.csv     # SMS spam classification dataset
│   └── requirements.txt         # Python dependencies
└── 3- DL/                       # Project 3: Deep Learning Deployment
    ├── *.ipynb                  # Deep learning training notebooks
    ├── app.py                   # FastAPI production application
    ├── Dockerfile               # Multi-stage container build
    ├── docker-compose.yml       # Orchestration configuration
    ├── sample_images/           # Test images for API validation
    └── test_report.md           # Comprehensive project validation
```

---

## Academic Standards and Quality Assurance

### Code Quality
All projects adhere to professional software development standards:
- **Documentation:** Comprehensive inline comments and README files
- **Error Handling:** Robust exception management and user feedback
- **Testing:** Validation scripts and health check systems
- **Security:** Best practices for production deployments

### Reproducibility
- **Environment Management:** Specified dependency versions and containers
- **Data Consistency:** Fixed datasets with documented characteristics  
- **Version Control:** Git-tracked development with clear commit history
- **Platform Independence:** Cross-platform compatibility (macOS, Linux, Windows)

---

## Future Extensions and Research Opportunities

### Advanced Topics Integration
- **AutoML and Hyperparameter Optimization:** Automated model selection and tuning
- **Model Interpretability:** LIME, SHAP, and explainable AI techniques
- **Advanced Architectures:** Transformer models, generative networks, reinforcement learning
- **Ethical AI:** Bias detection, fairness metrics, and responsible AI development
- **Distributed Computing:** Multi-GPU training and federated learning approaches

### Industry Connections
- **Professional Tools:** Integration with MLflow, Weights & Biases, Kubernetes
- **Cloud Platforms:** AWS, GCP, Azure deployment strategies
- **Production Monitoring:** Prometheus, Grafana, and observability practices
- **CI/CD Integration:** GitHub Actions, automated testing, and deployment pipelines

---

**Course Designer:** Faculty of Electrical and Computer Engineering  
**Last Updated:** August 2025  
**Maintenance:** Active development with industry best practice updates