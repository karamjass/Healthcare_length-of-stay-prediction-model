from src.data_cleaning import *
from src.feature_engineering import *
from src.modeling import *
from src.evaluation import *
import joblib

df = load_data("data/raw/train_data.csv")
df = remove_id_columns(df)
df = handle_missing_values(df)
df = handle_outliers(df)

df = create_high_utilizer_flag(df)
df = create_age_bins(df)
df = log_transform_features(df)

X, y = encode_features(df, target='Stay')
X_train, X_test, y_train, y_test = split_data(X, y)
X_train, X_test, scaler = scale_data(X_train, X_test)

model = train_xgboost(X_train, y_train)
evaluate_model(model, X_test, y_test)

joblib.dump(model, "models/xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")
