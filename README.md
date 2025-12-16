---
title: Weather Prediction API
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Weather Prediction MLOps Pipeline

A lightweight, production-ready MLOps system for weather prediction, integrating machine learning, workflow orchestration, automated CI/CD, containerization, and an interactive web interface — all deployed on free infrastructure.

https://alishbaramzan-weather-ml-api.hf.space/docs

# Overview

This project demonstrates a complete end-to-end MLOps workflow using open-source tools. It supports multiple ML tasks (classification, regression, clustering), exposes a FastAPI backend for real-time predictions, and provides a responsive web UI with dynamic weather animations and actionable recommendations.

# Features

- Multi-task ML: precipitation classification, temperature regression, clustering analysis
- FastAPI REST API with validation + Swagger docs
- Interactive Web UI (HTML, CSS, JS) with real-time predictions
- Prefect workflow for automated training
- Docker containerization for reproducible deployment
- GitHub Actions CI/CD for testing, building, and deployment
- Hugging Face Spaces hosting (free tier)

# Dataset

- Kaggle Weather History (96k hourly observations, 2006–2016) 
- Temporal features engineered (hour, day, month, season)
- Class imbalance handled with SMOTE
- Train/test split: 80/20 stratified

# Architecture
Data → Prefect → Training Pipeline → Model Artifacts
           ↓
      GitHub Actions →  Docker → Hugging Face Spaces
           ↓
        FastAPI → Web UI → End User

# Model Performance

- Classification: 98.63% accuracy, improved minority recall
- Regression: R² = 1.00
- Clustering: 4 meaningful weather patterns (K-Means)

# Local Setup

git clone https://github.com/alishbaaramzan/Weather-App-MLOPs.git

cd <repo>

docker build -t weather-app .

docker run -p 7860:7860 weather-app

Visit: http://localhost:7860

# Future Work

- Real-time weather data integration
- Ensemble models + uncertainty estimation
- Automated retraining + monitoring
- Multi-region forecasting
- Experiment tracking (MLflow/W&B)
