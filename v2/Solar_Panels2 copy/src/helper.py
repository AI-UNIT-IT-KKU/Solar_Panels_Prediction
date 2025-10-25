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
# 📊 Function 3: Outlier Summary Table
# ============================================
def plot_outliers(df, columns=None):
    """
    Generate a summary table of outliers (no plots).

    Parameters:
        df (pd.DataFrame): The DataFrame containing numerical data.
        columns (list, optional): Columns to analyze. If None, analyzes all numerical columns.

    Returns:
        pd.DataFrame: Table summarizing number and percentage of outliers per column.
    """
    import pandas as pd
    import numpy as np

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    summary = []

    for col in columns:
        series = df[col].dropna()

        # Compute IQR (Interquartile Range)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Define outlier limits
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Count outliers
        outliers = ((series < lower) | (series > upper)).sum()
        total = len(series)
        pct = (outliers / total * 100) if total > 0 else 0

        summary.append({
            "Column": col,
            "Outliers": int(outliers),
            "% of Total": round(pct, 3),
            "Lower Bound": round(lower, 3),
            "Upper Bound": round(upper, 3),
        })

    # Convert to DataFrame
    result = pd.DataFrame(summary).sort_values(by="% of Total", ascending=False).reset_index(drop=True)

    print("\n📊 Outlier Summary Report (IQR method):")
    print(result.to_string(index=False))

    return result
