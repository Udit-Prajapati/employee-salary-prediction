# File: app/streamlit_app.py

import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the education encoder
with open("../models/education_encoder.pkl", "rb") as f:
    education_encoder = pickle.load(f)

# Streamlit UI
st.title("💼 Employee Salary Prediction")
st.write("Enter employee details to predict if their salary is >50K or <=50K")

# Input fields
age = st.slider("Age", min_value=17, max_value=90, value=30)

# Use original class labels from the encoder
education_labels = list(education_encoder.classes_)
education_input = st.selectbox("Education Level", education_labels)

hours_per_week = st.slider("Hours Worked per Week", 1, 99, 40)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

# Predict button
if st.button("Predict"):
    # Encode education input
    try:
        encoded_education = education_encoder.transform([education_input])[0]
    except:
        st.error("Invalid education level selected.")
        st.stop()

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'hours-per-week': hours_per_week,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'education': encoded_education
    }])

    # Make prediction
    prediction = model.predict(input_data)
    result = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"💰 Predicted Salary: {result}")
