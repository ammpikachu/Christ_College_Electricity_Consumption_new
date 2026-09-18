import streamlit as st
import joblib

# Load the trained model
model = joblib.load("electricity_bill_new.pkl")

# Load the polynomial feature transformer
poly = joblib.load("polynomial_features (1).pkl")

# Page title
st.title("Electricity Consumption Prediction")

st.write("Enter the required details to predict electricity consumption.")

# Input fields
temperature = st.number_input(
    "Temperature",
    min_value=0.0,
    max_value=60.0,
    value=25.0
)

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

# Prediction button
if st.button("Predict"):

    # Create input data
    input_data = [[temperature, humidity]]

    # Transform input using polynomial features
    input_poly = poly.transform(input_data)

    # Make prediction
    prediction = model.predict(input_poly)

    # Display result
    st.success(f"Predicted Electricity Consumption: {prediction[0]:.2f}")
