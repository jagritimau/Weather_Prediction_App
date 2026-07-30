import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime # Import datetime module

# --- Load the trained model, scaler, and feature columns ---
# Ensure these files are in the same directory as your Streamlit app script
try:
    model = joblib.load('linear_regression_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    st.success("Model, scaler, and feature columns loaded successfully.")
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Please ensure they are in the same directory as this script and in your GitHub repository.")
    st.stop() # Stop the app if crucial files are missing
except Exception as e:
    st.error(f"An unexpected error occurred while loading files: {e}")
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Weather Temperature Predictor", layout="centered")
st.title("☀️ Weather Temperature Predictor")
st.write("Enter the weather conditions to predict the next day's temperature.")

# --- Input Features ---
st.header("Input Weather Conditions")

# Using a form to group inputs and trigger prediction on form submission
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
        wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)

        st.subheader("Previous Day's Temperature (for Lagged Features)")
        temp_lag_1d = st.number_input("Temperature 1 day ago (°C)", value=10.0, step=0.1)
        temp_lag_7d = st.number_input("Temperature 7 days ago (°C)", value=8.0, step=0.1)
        temp_lag_30d = st.number_input("Temperature 30 days ago (°C)", value=5.0, step=0.1)

    with col2:
        st.subheader("Rolling Statistics (from recent data)")
        temp_rolling_mean_7d = st.number_input("7-day Rolling Mean Temperature (°C)", value=9.0, step=0.1)
        temp_rolling_std_7d = st.number_input("7-day Rolling Std Temperature (°C)", value=2.0, step=0.1)

        prediction_date = st.date_input("Prediction Date", datetime.date(2023, 1, 10))

        # New: Sunrise and Sunset minutes input
        sunrise_hour = st.slider("Sunrise Hour", min_value=0, max_value=23, value=7)
        sunrise_minute = st.slider("Sunrise Minute", min_value=0, max_value=59, value=0)
        sunset_hour = st.slider("Sunset Hour", min_value=0, max_value=23, value=18)
        sunset_minute = st.slider("Sunset Minute", min_value=0, max_value=59, value=0)

        sunrise_minutes = sunrise_hour * 60 + sunrise_minute
        sunset_minutes = sunset_hour * 60 + sunset_minute

        # New: Weather Condition selection
        weather_condition_input = st.selectbox(
            "Weather Condition",
            ['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy', 'Foggy', 'Snowy'],
            index=0 # Default to Sunny
        )

    submitted = st.form_submit_button("Predict Temperature")

# --- Prediction Logic ---
if submitted:
    # Calculate Fourier features based on the selected date
    month = prediction_date.month
    dayofyear = prediction_date.timetuple().tm_yday
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear / 365)
    dayofyear_cos = np.cos(2 * np.pi * dayofyear / 365)

    # Prepare one-hot encoded weather conditions
    weather_conditions_ohe = {
        'Weather_Cloudy': 0, 'Weather_Foggy': 0, 'Weather_Partly Cloudy': 0,
        'Weather_Rainy': 0, 'Weather_Snowy': 0, 'Weather_Sunny': 0
    }
    if weather_condition_input: # Ensure it's not empty
        ohe_col_name = f"Weather_{weather_condition_input.replace(' ', '_')}"
        if ohe_col_name in weather_conditions_ohe:
            weather_conditions_ohe[ohe_col_name] = 1

    # Create a dictionary for the input data, ensuring all features are present
    input_data_dict = {
        'Humidity': humidity,
        'WindSpeed': wind_speed,
        'Temperature_Lag_1d': temp_lag_1d,
        'Temperature_Lag_7d': temp_lag_7d,
        'Temperature_Lag_30d': temp_lag_30d,
        'Temperature_Rolling_Mean_7d': temp_rolling_mean_7d,
        'Temperature_Rolling_Std_7d': temp_rolling_std_7d,
        'Month_sin': month_sin,
        'Month_cos': month_cos,
        'DayOfYear_sin': dayofyear_sin,
        'DayOfYear_cos': dayofyear_cos,
        'Sunrise_minutes': sunrise_minutes,
        'Sunset_minutes': sunset_minutes,
    }
    input_data_dict.update(weather_conditions_ohe)

    # Create a DataFrame from user inputs, ensuring correct column order and data types
    input_data = pd.DataFrame([input_data_dict])[feature_columns]

    # Scale the input data using the loaded scaler
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)[0]

    st.success(f"The predicted temperature for {prediction_date.strftime('%Y-%m-%d')} is: **{prediction:.2f} °C**")
    st.balloons()
