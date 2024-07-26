import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('trained_heart_lr_model.pkl')

st.title('Heart Disease Prediction')
st.write("This app predicts the likelihood of an individual having heart disease based on their health data.")


# Collect user input
smoking = st.radio('Do you smoke usually?', ['Yes', 'No'])
alcohol_drinking = st.radio('Do you drink alcohol usually?', ['Yes', 'No'])
stroke = st.radio('Have you ever had a stroke?', ['Yes', 'No'])
physical_health = st.slider('In the past 30 days, how many days was your physical health not good?', 0, 30, 15)
mental_health = st.slider('In the past 30 days, how many days was your mental health not good?', 0, 30, 15)
diff_walking = st.radio('Do you have serious difficulty walking or climbing stairs?', ['Yes', 'No'])
sex = st.radio('What is your sex?', ['Male', 'Female'])
diabetic = st.radio('Are you diabetic?', ['Yes', 'No'])
physical_activity = st.radio('Do you engage in physical activity?', ['Yes', 'No'])
asthma = st.radio('Do you have asthma?', ['Yes', 'No'])
kidney_disease = st.radio('Do you have kidney disease?', ['Yes', 'No'])
skin_cancer = st.radio('Do you have skin cancer?', ['Yes', 'No'])
sleep_time = st.slider('How many hours do you sleep on average per day?', 0, 24, 7)
age_category = st.selectbox('Select your age category', 
                            ['18-24', '25-29', '30-34', '35-39', '40-44', 
                             '45-49', '50-54', '55-59', '60-64', '65-69', 
                             '70-74', '75-79', '80 or older'])
race = st.selectbox('Select your race', 
                    ['American Indian/Alaskan Native', 'Asian', 'Black', 
                     'Hispanic', 'Other', 'White'])
gen_health = st.selectbox('How would you rate your general health?', 
                          ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
bmi_category = st.selectbox('What is your BMI category?', 
                            ['Underweight', 'Normal Weight', 'Overweight', 'Obese'])

# Convert Yes/No to 1/0
def convert_to_binary(value):
    return 1 if value == 'Yes' else 0

# Create a dictionary of the inputs
user_data = {
    'Smoking': convert_to_binary(smoking),
    'AlcoholDrinking': convert_to_binary(alcohol_drinking),
    'Stroke': convert_to_binary(stroke),
    'PhysicalHealth': physical_health,
    'MentalHealth': mental_health,
    'DiffWalking': convert_to_binary(diff_walking),
    'Sex': 1 if sex == 'Male' else 0,
    'Diabetic': convert_to_binary(diabetic),
    'PhysicalActivity': convert_to_binary(physical_activity),
    'Asthma': convert_to_binary(asthma),
    'KidneyDisease': convert_to_binary(kidney_disease),
    'SkinCancer': convert_to_binary(skin_cancer),
    'SleepTime': sleep_time,
    'AgeCategory_' + age_category: 1,
    'Race_' + race: 1,
    'GenHealth_' + gen_health: 1,
    'BMICategory_' + bmi_category: 1,
}

# Convert the dictionary to a DataFrame and fill missing columns with 0
input_df = pd.DataFrame([user_data])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Make predictions
if st.button('Predict'):
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]
    probability = round(prediction_proba * 100, 2)
    
    if prediction == 1:
        st.write(f'The model predicts that the individual has heart disease with a probability of {probability}%.')
    else:
        st.write(f'The model predicts that the individual does not have heart disease with a probability of {100 - probability}%.')
