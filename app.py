import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
from dotenv import load_dotenv

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.markdown("Using deployed MLflow stacking model")

# -------------------------------------------------------
# LOAD ENV VARIABLES (DagsHub MLflow Auth)
# -------------------------------------------------------
load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(
    "https://dagshub.com/JAINAM576/heart-disease-kaggle-ml-pipeline.mlflow"
)

# -------------------------------------------------------
# LOAD MODEL (cached)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model = mlflow.sklearn.load_model(
        "models:/stacking_heart_disease_model/1"
    )
    return model

model = load_model()

THRESHOLD = 0.4574  # PR curve threshold


# -------------------------------------------------------
# USER INPUT FORM
# -------------------------------------------------------
st.subheader("Enter Patient Details")

with st.form("prediction_form"):

    # -------- Numerical --------
    age = st.number_input("Age", min_value=20, max_value=100, value=55)
    bp = st.number_input("Systolic BP", min_value=80, max_value=220, value=130)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)

    # -------- Categorical --------
    sex = st.selectbox("Sex", [0, 1])
    chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    fbs = st.selectbox("FBS over 120", [0, 1])
    ekg = st.selectbox("EKG Results", [0, 1, 2])
    exercise_angina = st.selectbox("Exercise Angina", [0, 1])
    slope = st.selectbox("Slope of ST", [1, 2, 3])
    vessels = st.selectbox("Number of Vessels (Fluoroscopy)", [0, 1, 2, 3])
    thallium = st.selectbox("Thallium", [3, 6, 7])

    submit = st.form_submit_button("Predict")


# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------
if submit:

    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thallium
    }])

    # Decision score (raw)
    decision_score = model.decision_function(input_df)[0]

    # Convert to probability (sigmoid)
    probability = 1 / (1 + np.exp(-decision_score))

    # Apply custom PR-curve threshold
    prediction = 1 if probability >= THRESHOLD else 0

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease (Presence)")
    else:
        st.success("✅ Low Risk (Absence)")

    st.write(f"**Decision Score:** {decision_score:.4f}")
    st.write(f"**Risk Probability:** {probability:.4f}")
    st.write(f"**Threshold Used:** {THRESHOLD}")

    # Risk interpretation bar
    st.progress(float(probability))