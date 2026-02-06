import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ===============================
# Encode Features + Target
# ===============================
def encode_features(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ----- TARGET ENCODING (VERY IMPORTANT) -----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # ----- Ordinal Encoding -----
    if 'Severity of Illness' in X.columns:
        severity_map = {
            'Minor': 0,
            'Moderate': 1,
            'Severe': 2,
            'Extreme': 3
        }
        X['Severity of Illness'] = X['Severity of Illness'].map(severity_map)

    # ----- One-Hot Encoding -----
    X = pd.get_dummies(X, drop_first=True)

    return X, y


# ===============================
# Train-Test Split
# ===============================
def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )


# ===============================
# Scaling
# ===============================
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ===============================
# Model Training Functions
# ===============================
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ===============================
# MAIN PIPELINE
# ===============================
if __name__ == "__main__":

    from datacleaning import clean_data
    from feature_engineering import feature_engineering
    from evaluation import evaluate_all_models

    # Load & Clean Data
    df = clean_data("C:\\Users\\hp\\OneDrive\\Desktop\\healthcare project\\data\\raw\\healthcare\\train_data.csv")

    # Feature Engineering
    df = feature_engineering(df)

    # Encoding
    X, y = encode_features(df, target_col="Stay")

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scaling
    X_train, X_test, scaler = scale_data(X_train, X_test)

    print("✅ Encoding, splitting & scaling completed")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # ===============================
    # Train Models
    # ===============================
    print("\n🚀 Training Models...")

    lr = train_logistic_regression(X_train, y_train)
    print("✔ Logistic Regression trained")

    rf = train_random_forest(X_train, y_train)
    print("✔ Random Forest trained")

    xgb = train_xgboost(X_train, y_train)
    print("✔ XGBoost trained")

    cb = train_catboost(X_train, y_train)
    print("✔ CatBoost trained")

    # ===============================
    # Evaluation
    # ===============================
    models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb,
        "CatBoost": cb
    }

    print("\n📊 Evaluating Models...")
    results = evaluate_all_models(models, X_test, y_test)

    print("\n🏆 Final Model Comparison:")
    for model, metrics in results.items():
        print(model, "->", metrics)
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgboost.pkl")

