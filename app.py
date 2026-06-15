import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- APPLICATION LAYOUT ---
st.set_page_config(page_title="CardioPredict", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's metrics below to evaluate 10-year cardiovascular risk.")

# --- MINIMAL MODEL RE-TRAINING (Using the cleaned data) ---
@st.cache_resource
def load_trained_model():
    df = pd.read_csv('framingham.csv')
    df.drop(columns=['education'], inplace=True, errors='ignore')
    df.dropna(axis=0, inplace=True)

    X = df.drop(['TenYearCHD'], axis=1)
    y = df['TenYearCHD']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

model, scaler, feature_names = load_trained_model()

# --- INPUT FIELDS FOR THE USER ---
st.header("Patient Attributes Form")
col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    age = st.number_input("Age (Years)", min_value=1, max_value=100, value=45)
    currentSmoker = st.selectbox("Current Smoker?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0 if currentSmoker==0 else 10)
    BPMeds = st.selectbox("On Blood Pressure Medication?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    prevalentStroke = st.selectbox("History of Stroke?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    prevalentHyp = st.selectbox("History of Hypertension?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    diabetes = st.selectbox("Has Diabetes?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=50, max_value=600, value=220)
    sysBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
    diaBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, value=80)
    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=75)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=30, max_value=500, value=80)

# --- PREDICTION TRIGGER ---
if st.button("Calculate Cardiac Risk Profile", type="primary"):
    # Group inputs into an array matching the data structure
    input_data = np.array([[male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp,
                            diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale user data and make prediction
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    risk_percentage = probabilities[1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High Risk Profile:** The model calculates a **{risk_percentage:.1f}%** chance of heart disease development within 10 years.")
    else:
        st.success(f"✅ **Low Risk Profile:** The model calculates a **{risk_percentage:.1f}%** chance of heart disease development within 10 years.")
