import pandas as pd
import numpy as np

# ===============================
# Load Data
# ===============================
def load_data(path):
    """
    Load CSV data from given path
    """
    return pd.read_csv(path)


# ===============================
# Basic Integrity Check
# ===============================
def basic_integrity_check(df):
    """
    Prints basic info about the dataset
    """
    print("Dataset Shape:", df.shape)
    print("Duplicate Rows:", df.duplicated().sum())
    print("\nData Types:\n", df.dtypes)


# ===============================
# Drop ID / Non-predictive Columns
# ===============================
def drop_id_columns(df):
    """
    Drops columns containing 'id' in their name
    """
    df = df.copy()
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df.drop(columns=id_cols, inplace=True, errors='ignore')
    return df


# ===============================
# Handle Missing Values
# ===============================
def handle_missing_values(df):
    """
    - Drop columns with >50% missing
    - Numeric: median
    - Categorical: mode
    """
    df = df.copy()

    for col in df.columns:
        missing_pct = df[col].isnull().mean()

        if missing_pct > 0.50:
            df.drop(columns=[col], inplace=True)
        else:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    return df


# ===============================
# Handle Outliers (Winsorization)
# ===============================
def handle_outliers(df):
    """
    Caps numeric values at 95th percentile
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        upper_limit = df[col].quantile(0.95)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

    return df


# ===============================
# Run Cleaning Pipeline (OPTIONAL)
# ===============================
def clean_data(path):
    """
    Full data cleaning pipeline
    """
    df = load_data(path)
    basic_integrity_check(df)
    df = drop_id_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    return df


# ===============================
# Test Run
# ===============================
if __name__ == "__main__":
    df = load_data(r"C:\Users\hp\OneDrive\Desktop\healthcare project\data\raw\healthcare\train_data.csv")
    df = drop_id_columns(df)
    df = handle_missing_values(df)
    df = handle_outliers(df)
    print("✅ Data cleaning successful")
    print(df.shape)
