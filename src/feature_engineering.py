import pandas as pd
import numpy as np

# ===============================
# High Utilizer Flag
# ===============================
def create_high_utilizer_flag(df):
    df = df.copy()
    if 'Previous Admissions' in df.columns:
        df['High_Utilizer'] = (df['Previous Admissions'] > 3).astype(int)
    return df


# ===============================
# Log Transform Numeric Features
# ===============================
def log_transform_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if (df[col] >= 0).all():
            df[col] = np.log1p(df[col])

    return df


# ===============================
# Full Feature Engineering Pipeline
# ===============================
def feature_engineering(df):
    df = create_high_utilizer_flag(df)
    df = log_transform_features(df)
    return df
