# app.py

import streamlit as st
import joblib
import numpy as np
import os

# ---------------- LOAD MODEL & ENCODERS ----------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))

# ---------------- UI ----------------
st.title("🧠 Stroke Prediction App")
st.write("Enter patient details to predict stroke risk")

# ---------------- INPUT FIELDS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])

work_type = st.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
)

residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

avg_glucose_level = st.number_input(
    "Average Glucose Level", min_value=50.0, value=100.0
)

bmi = st.number_input(
    "BMI", min_value=10.0, value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    ["formerly smoked", "never smoked", "smokes", "Unknown"]
)

# ---------------- ENCODE INPUTS (SAME AS TRAINING) ----------------
gender = encoders['gender'].transform([gender])[0]
ever_married = encoders['ever_married'].transform([ever_married])[0]
work_type = encoders['work_type'].transform([work_type])[0]
residence_type = encoders['Residence_type'].transform([residence_type])[0]
smoking_status = encoders['smoking_status'].transform([smoking_status])[0]

# ---------------- PREDICTION ----------------
if st.button("Predict Stroke Risk"):
    input_data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        avg_glucose_level,
        bmi,
        smoking_status
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")
