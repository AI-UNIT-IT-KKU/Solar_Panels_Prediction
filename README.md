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

- **v3 - Daylight-Only Data Version**
Trains and evaluates models using only daylight samples — nighttime data was completely removed.
As a result, each season now contains almost half the number of samples compared to the previous version,
allowing a focused assessment of model accuracy and stability under daytime-only conditions.

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

---

## 🎥 Demo — Website in Action

Below is a short demonstration of the solar power prediction web interface.  
It showcases how users can input their data and instantly view the predicted power output.

<p align="center">
  <img src="https://github.com/AI-UNIT-IT-KKU/Solar_Panels_Prediction/raw/main/v3/Static/images/demo.gif" alt="Solar Power Prediction Demo" width="800">
</p>

---

## 🧠 Technologies Used

This project integrates multiple technologies to achieve accurate predictions and a user-friendly interface:

- 🐍 **Python** — Core programming language used for data analysis and model development.  
- 📓 **Jupyter Notebook** — Used for experimentation, model training, and visualization.  
- ⚙️ **XGBoost** — The main machine learning algorithm powering the solar power predictions.  
- 🚀 **FastAPI** — Backend framework for building the interactive web interface.  
- 🧮 **Pandas & NumPy** — For data preprocessing, cleaning, and numerical computations.  
- 📊 **scikit-learn** — Used for feature selection, evaluation metrics, and preprocessing utilities.  
- 🌐 **HTML, CSS, JavaScript** — To create the simple, responsive frontend connected to FastAPI.

---


