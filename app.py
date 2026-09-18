```python
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and polynomial transformer
model = joblib.load("electricity_bill.pkl")
poly = joblib.load("polynomial_features.pkl")

# Page title
st.title("⚡ Electricity Bill Prediction")

st.write(
    "Predict your electricity bill based on AC and Fan consumption "
    "using Polynomial Regression."
)

# -----------------------------
# AC Units Input
# -----------------------------

ac_units = st.number_input(
    "Enter AC Consumption (AC Units)",
    min_value=-100000.0,
    max_value=100000.0,
    value=100.0,
    step=1.0
)

# -----------------------------
# Fan Units Input
# -----------------------------

fan_units = st.number_input(
    "Enter Fan Consumption (Fan Units)",
    min_value=-100000.0,
    max_value=100000.0,
    value=50.0,
    step=1.0
)

# -----------------------------
# Validate Inputs
# -----------------------------

valid_ac = 0 <= ac_units <= 150
valid_fan = 0 <= fan_units <= 150

if not valid_ac or not valid_fan:

    st.error(
        "⚠️ Min value should be 0 and max value should be 150. "
        "Please enter a value within this range."
    )

else:

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("Predict Bill"):

        # Create DataFrame with the same feature names
        new_data = pd.DataFrame({
            "AC_Units": [ac_units],
            "Fan_Units": [fan_units]
        })

        # Convert input into polynomial features
        new_data_poly = poly.transform(new_data)

        # Predict electricity bill
        prediction = model.predict(new_data_poly)

        # Display success message
        st.success("Model predicted successfully!")

        # Display predicted bill
        st.metric(
            "Predicted Electricity Bill",
            f"₹{prediction[0]:,.2f}"
        )
```
