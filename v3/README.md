# ☀️ Solar Panels Power Prediction using XGBoost

This project predicts the power output of solar panels using XGBoost regression models, trained separately for each season — Winter, Spring, Summer, and Fall.  
All models are trained and tested **only on daylight data**, with nighttime samples fully removed to focus on realistic energy-producing conditions.  
As a result, each season now contains roughly **half the original dataset**, ensuring that training is based solely on valid daylight power readings.

By using XGBoost exclusively, the system achieves consistent high accuracy and efficiency across all datasets.  
The seasonal split enhances adaptability to real-world conditions, ensuring each model performs optimally for its respective time of year.

---

## ⚙️ Features

### ✅ Data Preprocessing & Cleaning
- Removes missing and duplicate records  
- Handles nighttime power values by removing all irradiance < 1 samples  
- Performs linear interpolation for continuous time-series data  
- Drops unnecessary time-related columns after processing  
- Automatically detects and merges duplicate timestamps by averaging numeric columns  

### ✅ Feature Selection
- Uses Pearson Correlation Coefficient (PCC) for each season to find the most influential features  
- Automatically detects and stores the model-used features  
- Ensures that only relevant features are used during training and prediction  
- Enhances interpretability and reduces noise in seasonal datasets  

### ✅ Model Training
- Trains an XGBoost Regressor for each season using optimized hyperparameters  
- Each model is trained on **daylight-only cleaned and preprocessed data**  
- Compares model performance across all seasons (Winter, Spring, Summer, Fall)  
- Saves the best trained model for each season as `xgb_{season}.joblib`  
- Uses consistent architecture and evaluation pipeline to ensure reliability  

### ✅ Evaluation Metrics
- R², MSE, RMSE per season  
- RMSE also expressed as a percentage of total system capacity  
- 5-fold cross-validation for stability and robustness  
- Detailed logging of model performance after each training phase  

### ✅ Web Deployment
- Interactive web interface built with FastAPI  
- User selects a season and inputs feature values to get real-time predictions  
- Clean, bilingual UI  
- Displays output in a clear format  

---

## 🧩 Helper Functions (`helper.py`)

| Function | Description |
|-----------|-------------|
| `fill_missing_values(df, strategy='mean')` | Replaces NaN values using mean, median, or constant strategies |
| `drop_missing_values(df, columns=None)` | Drops rows with missing values in selected columns |
| `plot_outliers(df, columns=None)` | Detects and visualizes outliers using IQR or boxplots |

---

**📘 Version:** v3 — Daylight-Only Data Version  
🌞 Trained and evaluated only on daylight samples (nighttime removed).

