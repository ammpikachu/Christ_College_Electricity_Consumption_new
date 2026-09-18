import gradio as gr
import joblib

# Load the trained model
model = joblib.load("electricity_bill_new.pkl")

# Load polynomial feature transformer
poly = joblib.load("polynomial_features (1).pkl")


def predict_electricity(ac_units, fan_units):
    try:
        # Prepare input
        input_data = [[ac_units, fan_units]]

        # Transform using polynomial features
        input_poly = poly.transform(input_data)

        # Predict
        prediction = model.predict(input_poly)

        return f"Predicted Electricity Consumption: {prediction[0]:.2f}"

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
app = gr.Interface(
    fn=predict_electricity,
    inputs=[
        gr.Number(
            label="AC Units",
            minimum=0,
            precision=0
        ),
        gr.Number(
            label="Fan Units",
            minimum=0,
            precision=0
        )
    ],
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="Electricity Consumption Prediction",
    description="Enter the number of AC and Fan units to predict electricity consumption."
)

# Start Gradio
app.launch(
    server_name="0.0.0.0",
    server_port=7860
)
