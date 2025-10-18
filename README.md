# ☀️ Solar Panels Power Prediction using XGBoost
This script trains an optimized XGBoost regression model to predict solar panel power output using selected feature sets (IG and PCC). It includes automatic hyperparameter tuning via RandomizedSearchCV, missing value handling, performance evaluation (R², RMSE), and visualizations for model results and feature importance.  

This project aims to **predict the generated power output of solar panels** at King Khalid University using advanced machine learning techniques — specifically the **XGBoost regression model**.  
The project includes full data cleaning, preprocessing, feature selection, model training, and API deployment using **FastAPI**.

---


---

## ⚙️ Features

✅ **Data Preprocessing & Cleaning**
- Removes missing and duplicate records  
- Handles nighttime power values by setting negatives to zero  
- Performs **linear interpolation** for continuous time-series data  
- Drops unnecessary time-related columns after processing  

✅ **Feature Selection**
- Uses **Information Gain (IG)** and **Pearson Correlation Coefficient (PCC)**  
- Selects the most influential features for solar power prediction  

✅ **Model Training**
- Trains an **XGBoost Regressor** using optimized hyperparameters (via RandomizedSearchCV)
- Compares IG and PCC feature sets for the best model performance  
- Saves the best trained model (`xgb_best_pcc.joblib`)  

✅ **Evaluation Metrics**
- R², MSE, RMSE  
- RMSE as a percentage of total system capacity  
- Stability check using **5-fold cross-validation**  

✅ **Web Deployment**
- Interactive web interface built with **FastAPI**
- User inputs features to get real-time predictions  
- Clean, responsive UI with dual sections (landing & prediction)  

---
#🧩 Helper Functions (helper.py)
| Function                                   | Description                                                    |
| ------------------------------------------ | -------------------------------------------------------------- |
| `fill_missing_values(df, strategy='mean')` | Replaces NaN values using mean, median, or constant strategies |
| `drop_missing_values(df, columns=None)`    | Drops rows with missing values in selected columns             |
| `plot_outliers(df, columns=None)`          | Detects and visualizes outliers using IQR or boxplots          |


---
