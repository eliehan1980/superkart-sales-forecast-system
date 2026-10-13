

# Import NumPy for array manipulations and numeric handling
import numpy as np

# Import joblib to deserialize and load the trained pipeline artifact
import joblib

# Import pandas to convert inbound prediction payloads into structured DataFrames
import pandas as pd

# Import core Flask components for app initialization, request handling, and JSON serialization
from flask import Flask, request, jsonify

# Instantiate the Flask web application with a descriptive application name
superkart_api = Flask("SuperKart")

# Deserialize and load the trained Random Forest pipeline artifact into memory at startup
model = joblib.load("superkart_model.joblib")

# Define a root GET endpoint to serve as a basic homepage health-check route
@superkart_api.get('/')
def home():
    # Return a status confirmation message verifying the API service is reachable
    return "Welcome to the SuperKart System"

# Define a POST endpoint for real-time single-record sales predictions
@superkart_api.post('/v1/predict')
def predict_sales():
    # Parse the incoming JSON payload sent in the request body
    data = request.get_json()

    # Construct a dictionary mapping each input attribute to match the model training schema
    sample = {
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        'Product_Allocated_Area': data['Product_Allocated_Area'],
        'Product_MRP': data['Product_MRP'],
        'Store_Size': data['Store_Size'],
        'Store_Location_City_Type': data['Store_Location_City_Type'],
        'Store_Type': data['Store_Type'],
        'Product_Id_char': data['Product_Id_char'],
        'Store_Age_Years': data['Store_Age_Years'],
        'Product_Type_Category': data['Product_Type_Category']
    }

    # Wrap the single dictionary into a single-row pandas DataFrame expected by the pipeline
    input_data = pd.DataFrame([sample])

    # Pass the DataFrame through the pipeline to preprocess features and generate prediction
    prediction = model.predict(input_data).tolist()[0]

    # Return the forecast formatted as a JSON response
    return jsonify({'Sales': prediction})

# Define a POST endpoint to handle batch inference via uploaded CSV files
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Retrieve the uploaded CSV file from the multipart form request
    file = request.files['file']

    # Read the uploaded CSV data directly into a pandas DataFrame
    input_data = pd.read_csv(file)

    # Run vectorized inference on the entire batch and convert NumPy array output to a list
    predictions = model.predict(input_data).tolist()

    # Map array indices to their predicted sales rounded to 2 decimal places
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    # Return the indexed prediction dictionary as a JSON response
    return output_dict

# Standard entry point to start the local Flask development server when executed directly
if __name__ == '__main__':
    # Launch the server with live reload and debugging active
    superkart_api.run(debug=True)
