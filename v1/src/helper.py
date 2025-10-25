# ============================================
# 📦 helper.py — Utility functions for data cleaning & visualization
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# ============================================
# 🧩 Function 1: Fill Missing Values 
# ============================================
def fill_missing_values(df, strategy='mean', use_interpolation=False):
    """
    Fill missing values in a DataFrame using either SimpleImputer or linear interpolation.

    Parameters:
        df (pd.DataFrame): The DataFrame containing missing values.
        strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        use_interpolation (bool): If True, use linear interpolation instead of SimpleImputer.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if use_interpolation:
        # ✅ Use linear interpolation — best for time-series data such as solar energy readings.
        df_imputed = df.interpolate(method='linear', limit_direction='both')
        print("🔁 Filled missing values using linear interpolation.")
        return df_imputed
    else:
        # 🧮 Use SimpleImputer for non-time-series datasets (mean/median/statistical filling).
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        print(f"📊 Filled missing values using SimpleImputer with strategy='{strategy}'.")
        return df_imputed


# ============================================
# 🧹 Function 2: Drop Missing Values
# ============================================
def drop_missing_values(df, columns=None):
    """
    Drop rows containing missing values.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        columns (list, optional): Specific columns to check. If None, checks all columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame without missing rows.
    """
    return df.dropna(subset=columns) if columns else df.dropna()


# ============================================
# 📊 Function 3: Plot Outliers
# ============================================
def plot_outliers(df, columns=None):
    """
    Visualize outliers in the given DataFrame columns using boxplots.

    Parameters:
        df (pd.DataFrame): The DataFrame containing numerical data.
        columns (list, optional): Columns to visualize. If None, plots all numerical columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    plt.figure(figsize=(10, 5 * len(columns)))

    for i, col in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Outliers in "{col}"', fontsize=12)
        plt.xlabel('')
    
    plt.tight_layout()
    plt.show()