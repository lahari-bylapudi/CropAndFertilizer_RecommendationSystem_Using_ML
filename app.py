import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained models
try:
    with open('crop_model.pkl', 'rb') as file:
        crop_model = pickle.load(file)
except FileNotFoundError:
    st.error("Crop model file not found.")
    st.stop()

try:
    with open('fertilizer_model.pkl', 'rb') as file:
        fertilizer_model = pickle.load(file)
except FileNotFoundError:
    st.error("Fertilizer model file not found.")
    st.stop()

# Streamlit UI
st.title("Crop and Fertilizer Recommendation System")

# Input fields
st.sidebar.header("Enter Soil and Climate Parameters")

# Crop Inputs
st.sidebar.subheader("Crop Prediction Inputs")
nitrogen_crop = st.sidebar.number_input("Nitrogen Content (N)", min_value=0, max_value=200, value=50)
phosphorus_crop = st.sidebar.number_input("Phosphorus Content (P)", min_value=0, max_value=200, value=50)
potassium_crop = st.sidebar.number_input("Potassium Content (K)", min_value=0, max_value=200, value=50)
temperature_crop = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity_crop = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
pH_crop = st.sidebar.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Fertilizer Inputs
st.sidebar.subheader("Fertilizer Recommendation Inputs")
temperature_fert = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, key="temp_fert")
humidity_fert = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, key="hum_fert")
moisture_fert = st.sidebar.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, key="mois_fert")
soil_type_fert = st.sidebar.number_input("Soil Type (Numerical)", min_value=0, max_value=5, value=0, key="soil_fert")
crop_type_fert = st.sidebar.number_input("Crop Type (Numerical)", min_value=0, max_value=20, value=0, key="crop_fert")
nitrogen_fert = st.sidebar.number_input("Nitrogen Content (N)", min_value=0, max_value=200, value=50, key="n_fert")
phosphorus_fert = st.sidebar.number_input("Phosphorus Content (P)", min_value=0, max_value=200, value=50, key="p_fert")
potassium_fert = st.sidebar.number_input("Potassium Content (K)", min_value=0, max_value=200, value=50, key="k_fert")

# Crop and Fertilizer Dictionaries (same as before)
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
    6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
    11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple',
    17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'
}

fert_dict = {
    1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 5: '17-17-17', 6: '20-20', 7: '10-26-26'
}


# Predict Crop
if st.sidebar.button("Predict Crop"):
    try:
        input_data_crop = pd.DataFrame([[nitrogen_crop, phosphorus_crop, potassium_crop, temperature_crop, humidity_crop, pH_crop, rainfall]],
                                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        st.write("Input Data for Crop Prediction:")
        st.write(input_data_crop)
        predicted_crop_index = crop_model.predict(input_data_crop)[0]
        predicted_crop_name = crop_dict.get(predicted_crop_index + 1)  # Correct mapping
        if predicted_crop_name:
            st.success(f"Recommended Crop: ['{predicted_crop_name}'] is a best crop to grow in the farm.")
        else:
            st.error("Invalid crop prediction. Check model or input data.")
    except Exception as e:
        st.error(f"An unexpected error occurred during crop prediction: {e}")

# Predict Fertilizer
if st.sidebar.button("Suggest Fertilizer"):
    try:
        input_data_fert = pd.DataFrame([[temperature_fert, humidity_fert, moisture_fert, soil_type_fert, crop_type_fert, nitrogen_fert, phosphorus_fert, potassium_fert]],
                                      columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
        st.write("Input Data for Fertilizer Prediction:")
        st.write(input_data_fert)
        predicted_fertilizer_index = fertilizer_model.predict(input_data_fert)[0]
        predicted_fertilizer_name = fert_dict.get(predicted_fertilizer_index + 1)  # Correct mapping
        if predicted_fertilizer_name:
            st.success(f"Recommended Fertilizer: ['{predicted_fertilizer_name}'] is a best fertilizer for the given conditions.")
        else:
            st.error("Invalid fertilizer prediction. Check model or input data.")
    except Exception as e:
        st.error(f"An unexpected error occurred during fertilizer prediction: {e}")