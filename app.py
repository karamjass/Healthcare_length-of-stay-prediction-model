import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Fix path
sys.path.append(os.path.abspath("src"))

from feature_engineering import feature_engineering
from modeling import encode_features


@st.cache_resource
def load_model():
    return joblib.load("models/xgboost.pkl")

model = load_model()


st.title("🏥 Hospital Length of Stay Prediction")

age = st.selectbox("Age Group",
["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80","81-90","91-100"])

severity = st.selectbox("Severity",
["Minor","Moderate","Severe","Extreme"])

admission_type = st.selectbox("Admission Type",
["Urgent","Trauma","Emergency"])

visitors = st.slider("Visitors",0,20,2)
previous = st.slider("Previous Admissions",0,10,1)


if st.button("Predict"):

    input_df = pd.DataFrame({
        "Age":[age],
        "Severity of Illness":[severity],
        "Admission Type":[admission_type],
        "Visitors with Patient":[visitors],
        "Previous Admissions":[previous]
    })

    input_df = feature_engineering(input_df)

    X,_ = encode_features(input_df, target_col=None)

    pred = model.predict(X)[0]

    st.success(f"Predicted Stay: {pred}")
