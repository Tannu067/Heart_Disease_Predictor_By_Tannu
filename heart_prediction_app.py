import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model, scaler, and columns
model = joblib.load("best_heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Streamlit app UI
st.title("💓 Heart Disease Predictor by Tannu Kumari")
st.markdown("Predict if a person is likely to have heart disease based on medical inputs.")

# User input form
with st.form("heart_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["asymptomatic", "nonanginal", "nontypical", "typical"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["fixed", "normal", "reversable"])

    submit = st.form_submit_button("Predict")

if submit:
    # Manual binary encoding
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # One-hot encode ChestPain
    cp_dict = {
        "ChestPain_nonanginal": cp == "nonanginal",
        "ChestPain_nontypical": cp == "nontypical",
        "ChestPain_typical": cp == "typical"
    }

    # One-hot encode Thal
    thal_dict = {
        "Thal_normal": thal == "normal",
        "Thal_reversable": thal == "reversable"
    }

    # Combine all into one dictionary
    input_dict = {
        "Age": age,
        "Sex": sex,
        "RestBP": trestbps,
        "Chol": chol,
        "Fbs": fbs,
        "RestECG": restecg,
        "MaxHR": thalach,
        "ExAng": exang,
        "Oldpeak": oldpeak,
        "Slope": slope,
        "Ca": ca,
        **cp_dict,
        **thal_dict
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure all expected columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! (Probability: {prob:.2f})")
    else:
        st.success(f"💚 Low Risk of Heart Disease! (Probability: {prob:.2f})")

