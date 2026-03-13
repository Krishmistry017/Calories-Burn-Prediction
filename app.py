import streamlit as st
import pandas as pd
import joblib

# Set page configuration
st.set_page_config(
    page_title="Fitness Calorie Predictor",
    page_icon="🔥",
    layout="wide"
)
st.title("🔥 Fitness Calorie Burn Predictor")
st.write("Enter your workout details to estimate calories burned.")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age",0, 1,100)
    Height = st.slider("Height (cm)", 0,140, 250)
    Gender = st.selectbox("Gender", ["Male","Female"])

with col2:
    Heart_Rate = st.slider("Heart Rate", 60, 200, 90)
    Body_Temp = st.slider("Body Temperature (°F)", 90.0, 110.0, 98.6)
# Add sidebar information
st.sidebar.title("About App")

st.sidebar.write(
"""
This machine learning model predicts calories burned during exercise
based on physical and workout data.
"""
)

st.sidebar.write("Model: XGBoost")

# Load trained model and scaler
model = joblib.load("calorie_model.pkl")
scaler = joblib.load("scaler.pkl")



if Gender == "Male":
    Gender = 0
else:
    Gender = 1

feature_names = ["Gender", "Age", "Height", "Heart_Rate", "Body_Temp"]

if st.button("Predict Calories"):

    user_data = pd.DataFrame([[Gender, Age, Height, Heart_Rate, Body_Temp]],
                         columns=feature_names)
    
    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)

    st.metric("Calories Burned", f"{prediction[0]:.2f} kcal")

    st.success("🔥 Great workout!")

