# In-Class Projects

This directory contains three hands-on projects designed to provide practical experience with real-world machine learning workflows.

## Project 1 – Exploratory Data Analysis
**Timing**: After the "Data" block  
**Theme**: Exploratory Data Analysis on the Student-Performance dataset  
**Goal**: Walk every student from raw CSV to clear, visual insights in a single notebook.

### Session Structure (90 minutes)

#### Kick-off (0–10 min)
- Briefly frame what "good" EDA looks like
- Outline the notebook checkpoints

#### Instructor live-coding demo (10–55 min)
- Load the dataset, preview types, and inspect missing values with a heat-map
- Do a quick categorical vs. numeric audit and create one engineered feature (e.g., study-time bands)
- Generate a pair-plot and surface the top three correlations that might influence final grades

#### Guided mini-exercise (55–80 min)
- Students stay in their own notebook and answer one of two prompts:
  - "Which non-academic factors correlate most with final grade?"
  - "How does study time interact with parental education?"
- Instructor roams to unblock

#### Lightning recap (80–90 min)
- Two volunteers project their plots
- Highlight best practices and common pitfalls

---

## Project 2 – ML Microservice
**Timing**: After Supervised + Unsupervised chapters  
**Theme**: Ship a classic ML micro-service (SMS-Spam classifier)  
**Goal**: Train, wrap, and containerize a simple model, then hit it with Postman inside 90 minutes.

### Session Structure (90 minutes)

#### Architecture walk-through (0–10 min)
- Explain the flow from notebook → model.joblib → FastAPI → Docker

#### Model training (10–35 min)
- Vectorize with TF-IDF, train Logistic Regression, evaluate quickly, and persist the artifact

#### API layer (35–60 min)
- Start from a FastAPI skeleton, add a /predict route that loads the model and returns JSON
- Test locally with the provided Postman collection

#### Containerization (60–80 min)
- Build the image, run `docker run -p 8000:80 spam-api`, and confirm the endpoint still works

#### Show-and-tell (80–90 min)
- One team demonstrates a live request/response
- Optionally push the image to Docker Hub for later use

**Prep**: Students clone the ml-api-starter repo and skim a two-page FastAPI cheat-sheet the night before.

---

## Project 3 – Deep Learning Deployment
**Timing**: After DL + Validation chapters  
**Theme**: Local Deep-Learning model to a live Hetzner endpoint  
**Goal**: Train a lightweight CNN locally, package it, and deploy a functional inference API on your Hetzner VM.

### Session Structure (90 minutes)

#### Kick-off (0–5 min)
- Clarify that all training happens on students' machines (or Colab) and only the container gets shipped

#### Local training (5–35 min)
- Use a pre-written notebook that trains a 6-layer CNN on CIFAR-10 (or Fashion-MNIST for CPU), three epochs, exports model.pt

#### API scaffolding (35–55 min)
- Reuse the FastAPI skeleton; add /predict_image and /health routes
- Unit-test with a sample JPEG to verify correct class prediction

#### Docker build and tag (55–75 min)
- Multi-stage build with the Torch runtime; check that inference works inside the container

#### Remote deploy (75–90 min)
- Students scp or docker push the image, SSH into Hetzner, run `docker compose up -d`, and ping the public URL from Postman

**Prep**: Provide a dl-deploy-starter repo with the training notebook and API skeleton, plus pre-provisioned SSH keys and a Hetzner VM that already has Docker CE and Compose installed.
