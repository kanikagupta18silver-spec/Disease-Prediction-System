import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# LOAD SAVED FILES
# =========================

model = joblib.load("models/heart_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
training_columns = joblib.load("models/training_columns.pkl")

# =========================
# APP TITLE
# =========================

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient health details below to predict heart disease risk.")

# =========================
# USER INPUTS
# =========================

age = st.number_input("Age", min_value=1, max_value=100, value=50)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200
)

thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

ca = st.number_input(
    "Number of Major Vessels",
    min_value=0,
    max_value=4,
    value=0
)

# =========================
# CATEGORICAL INPUTS
# =========================

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    ["True", "False"]
)

restecg = st.selectbox(
    "Rest ECG",
    [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ]
)

exang = st.selectbox(
    "Exercise Induced Angina",
    ["True", "False"]
)

slope = st.selectbox(
    "Slope",
    [
        "upsloping",
        "flat",
        "downsloping"
    ]
)

thal = st.selectbox(
    "Thal",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)

dataset = st.selectbox(
    "Dataset",
    [
        "Cleveland",
        "Hungary",
        "Switzerland",
        "VA Long Beach"
    ]
)

# =========================
# PREDICT BUTTON
# =========================

if st.button("Predict"):

    # Create input dictionary
    input_data = {
        "id": 1,
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)

    # Match training columns
    input_encoded = input_encoded.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)

    # =========================
    # OUTPUT
    # =========================

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(
        f"Prediction Confidence: {np.max(probability) * 100:.2f}%"
    )
