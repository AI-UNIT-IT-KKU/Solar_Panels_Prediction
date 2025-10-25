# ================================
# This code checks the similarity and validity of columns in a dataset
#=================================

import sys
import pandas as pd
import numpy as np
import joblib

# =============================================
# 🧩 Compatibility patch for older pandas objects
# Some old .pkl files contain outdated index types not in new pandas
# This block recreates those classes to allow successful loading
# =============================================

# Define replacements for older numeric index classes
class Int64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None):
        return pd.Index(data, dtype='int64', copy=copy, name=name)

class Float64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None):
        return pd.Index(data, dtype='float64', copy=copy, name=name)

class UInt64Index(pd.Index):
    def __new__(cls, data=None, dtype=None, copy=False, name=None):
        return pd.Index(data, dtype='uint64', copy=copy, name=name)

# Combine these into a dummy numeric index module to mimic the old structure
class NumericIndexModule:
    Int64Index = Int64Index
    Float64Index = Float64Index
    UInt64Index = UInt64Index
    RangeIndex = pd.RangeIndex

# Register the fake module so joblib can find it when loading older data
sys.modules['pandas.core.indexes.numeric'] = NumericIndexModule()

# Patch datetime and timedelta index modules (for older pandas versions)
class DatetimeIndexModule:
    DatetimeIndex = pd.DatetimeIndex

class TimedeltaIndexModule:
    TimedeltaIndex = pd.TimedeltaIndex

sys.modules['pandas.core.indexes.datetimes'] = DatetimeIndexModule()
sys.modules['pandas.core.indexes.timedeltas'] = TimedeltaIndexModule()

# =============================================
# 🧾 Try loading the dataset safely with joblib
# =============================================
try:
    df = joblib.load('Data/df1.pkl')
    print(f" SUCCESS!")  # Print success message if loading works
    print(f"Type: {type(df)}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))  # Display first 10 rows for a quick preview
    print(f"\nColumn names:")
    print(df.columns.tolist())  # List all column names
    print(f"\nDataFrame Info:")
    df.info()  # Display column data types and non-null counts
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()  # Print full error traceback for debugging

# =============================================
# 🔍 Missing Values Analysis
# =============================================

print("\n🔍 Missing Values Summary:")
missing_values = df.isnull().sum()  # Count missing (NaN) values per column
missing_percentage = (missing_values / len(df)) * 100  # Calculate percentage of missing data

# Combine results into a summary table
missing_report = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentage.round(2)
})

# Display only columns with missing values
print(missing_report[missing_report['Missing Values'] > 0])

# Optional: print all columns instead (even those with 0 missing values)
# print(missing_report)

# =============================================
# 🔢 Duplicate Value Check per Column
# =============================================
print("\n🔢 Duplicate Counts per Column:")
# Apply a lambda to each column to count how many values are repeated
print(df.apply(lambda x: x.duplicated().sum()))
