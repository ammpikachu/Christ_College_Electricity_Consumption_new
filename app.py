import streamlit as st
import joblib

# Load model and polynomial features
model = joblib.load("electricity_bill_new.pkl")
poly = joblib.load("polynomial_features (1).pkl")

# Page title
st.title("Electricity Consumption Prediction")

st.write("Enter the required details to predict electricity consumption.")

# AC input
ac = st.number_input(
    "AC Usage (hours)",
    min_value=0.0,
    max_value=24.0,
    value=0.0,
    step=1.0
)

# Fan input
fan = st.number_input(
    "Fan Usage (hours)",
    min_value=0.0,
    max_value=24.0,
    value=0.0,
    step=1.0
)

# Predict
if st.button("Predict"):

    # Input data
    input_data = [[ac, fan]]

    # Apply polynomial transformation
    input_poly = poly.transform(input_data)

    # Prediction
    prediction = model.predict(input_poly)

    # Display result
    st.success(
        f"Predicted Electricity Consumption: {prediction[0]:.2f}"
    )
