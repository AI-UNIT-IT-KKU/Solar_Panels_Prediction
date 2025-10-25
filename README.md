# ☀️ Solar Panels Power Prediction — Project Overview

This repository focuses on predicting solar panel power output using data-driven machine learning techniques.  
It contains **three main versions (v1, v2, v3)**, each representing a stage of model experimentation and improvement.

---

## 🔁 Versions Summary

- **v1 — Multi-Model Comparison**  
  Trains and compares several regression models to identify the most accurate one for solar power prediction.  
  After evaluation, **XGBoost** was found to be the best-performing model and is used in later versions.

- **v2 — Seasonal Model Training**  
  Splits the dataset into four seasons (**Winter, Spring, Summer, Fall**) and trains a separate XGBoost model for each.  
  Each seasonal model is trained on **100,000 data samples** to capture unique environmental conditions.

- **v3 — Data Reduction Test**  
  Repeats the seasonal training setup but reduces the training data size to **70,000 samples per season**  
  to observe how data volume affects performance and stability.

---

## ⚙️ Shared Core Across All Versions
All versions share the same foundational components:
- Data cleaning and preprocessing (handling missing values, duplicates, and outliers)  
- Consistent feature extraction and selection pipeline  
- Unified evaluation metrics (R², MSE, RMSE)  
- Reproducible training and testing structure  
- Organized workflow for future scalability

---

## 🎯 Purpose
The project aims to explore how different modeling strategies, seasonal segmentation, and dataset sizes  
influence the accuracy and reliability of solar power prediction using **XGBoost** as the core algorithm.

---


> Each version folder includes its own README with detailed steps, code explanations.

---

## 🧩 Overview
This repository represents an iterative journey — from model selection to optimization and efficiency testing —  
all focused on understanding how **machine learning** can improve solar energy forecasting.
