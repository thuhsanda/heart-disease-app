import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('trained_heart_lr_model.pkl')

# Define the app
st.title("Heart Disease Prediction")

st.write("This app predicts the likelihood of an individual having heart disease based on their health data.")

# Input fields for user data
smoking = st.selectbox("Do you smoke usually?", ("Yes", "No"))
alcohol = st.selectbox("Do you drink alcohol usually?", ("Yes", "No"))
stroke = st.selectbox("Have you had a stroke?", ("Yes", "No"))
physical_health = st.slider("Physical Health (Number of days in the past month)", 0, 30, 15)
mental_health = st.slider("Mental Health (Number of days in the past month)", 0, 30, 15)
diff_walking = st.selectbox("Do you have difficulty walking?", ("Yes", "No"))
sex = st.selectbox("Sex", ("Male", "Female"))
diabetic = st.selectbox("Are you diabetic?", ("Yes", "No"))
physical_activity = st.selectbox("Do you engage in physical activity?", ("Yes", "No"))
asthma = st.selectbox("Do you have asthma?", ("Yes", "No"))
kidney_disease = st.selectbox("Do you have kidney disease?", ("Yes", "No"))
skin_cancer = st.selectbox("Do you have skin cancer?", ("Yes", "No"))
sleep_time = st.slider("How many hours do you sleep per night?", 1, 24, 7)
age_category = st.selectbox("Age Category", ("18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"))
bmi_category = st.selectbox("BMI Category", ("Underweight", "Normal weight", "Overweight", "Obese"))
gen_health = st.selectbox("General Health", ("Excellent", "Very good", "Good", "Fair", "Poor"))

# Convert inputs to model-ready format
data = {
    'Smoking': 1 if smoking == 'Yes' else 0,
    'AlcoholDrinking': 1 if alcohol == 'Yes' else 0,
    'Stroke': 1 if stroke == 'Yes' else 0,
    'PhysicalHealth': physical_health,
    'MentalHealth': mental_health,
    'DiffWalking': 1 if diff_walking == 'Yes' else 0,
    'Sex': 1 if sex == 'Male' else 0,
    'Diabetic': 1 if diabetic == 'Yes' else 0,
    'PhysicalActivity': 1 if physical_activity == 'Yes' else 0,
    'Asthma': 1 if asthma == 'Yes' else 0,
    'KidneyDisease': 1 if kidney_disease == 'Yes' else 0,
    'SkinCancer': 1 if skin_cancer == 'Yes' else 0,
    'SleepingTime': sleep_time,
    'AgeCategory': age_category,
    'BMICategory': bmi_category,
    'GenHealth': gen_health
}

input_df = pd.DataFrame([data])

# Make prediction
prediction_prob = model.predict_proba(input_df)[:, 1][0]
prediction = "have heart disease" if prediction_prob > 0.5 else "not have heart disease"

# Display result
st.write(f"The individual is predicted to {prediction}.")
st.write(f"Prediction confidence: {prediction_prob * 100:.2f}%")

if st.button("Show Feature Importance"):
    st.write("Feature importances are:")
    importances = pd.Series(model.coef_[0], index=input_df.columns)
    st.bar_chart(importances)
