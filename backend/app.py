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


# Define a root GET endpoint to serve as a basic homepage route
@superkart_api.get("/")
def home():
  return "Welcome to the SuperKart Sales Forecasting System"


# Dedicated health-check route returning JSON status 200
@superkart_api.get("/health")
def health():
  return jsonify({"status": "healthy", "service": "SuperKart_Backend"}), 200


# Define a POST endpoint for real-time single-record sales predictions
@superkart_api.post("/v1/predict")
def predict_sales():
  # Parse the incoming JSON payload sent in the request body
  data = request.get_json()
  if not data:
    return (
        jsonify({"error": "Invalid request body. JSON payload expected."}),
        400,
    )

  # Check that all required keys are present before dictionary lookup
  required_keys = [
      "Product_Weight",
      "Product_Sugar_Content",
      "Product_Allocated_Area",
      "Product_MRP",
      "Store_Size",
      "Store_Location_City_Type",
      "Store_Type",
      "Product_Id_char",
      "Store_Age_Years",
      "Product_Type_Category",
  ]
  missing_keys = [key for key in required_keys if key not in data]
  if missing_keys:
    return (
        jsonify({
            "error": "Missing required schema fields",
            "missing_keys": missing_keys,
        }),
        400,
    )

  # Construct a dictionary mapping each input attribute to match the model training schema
  sample = {
      "Product_Weight": data["Product_Weight"],
      "Product_Sugar_Content": data["Product_Sugar_Content"],
      "Product_Allocated_Area": data["Product_Allocated_Area"],
      "Product_MRP": data["Product_MRP"],
      "Store_Size": data["Store_Size"],
      "Store_Location_City_Type": data["Store_Location_City_Type"],
      "Store_Type": data["Store_Type"],
      "Product_Id_char": data["Product_Id_char"],
      "Store_Age_Years": data["Store_Age_Years"],
      "Product_Type_Category": data["Product_Type_Category"],
  }

  # Wrap the single dictionary into a single-row pandas DataFrame expected by the pipeline
  input_data = pd.DataFrame([sample])

  # Pass the DataFrame through the pipeline to preprocess features and generate prediction
  prediction = model.predict(input_data).tolist()[0]

  # Return the forecast formatted as a JSON response
  return jsonify({"Sales": round(prediction, 2)})


# Define a POST endpoint to handle batch inference via uploaded CSV files
@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
  # The exact columns the model was trained on
  required_cols = [
      "Product_Weight",
      "Product_Sugar_Content",
      "Product_Allocated_Area",
      "Product_MRP",
      "Store_Size",
      "Store_Location_City_Type",
      "Store_Type",
      "Product_Id_char",
      "Store_Age_Years",
      "Product_Type_Category",
  ]
  # Helper that builds the standard "CSV not compatible" error response
  def csv_error(reason):
    return jsonify({
        "error": "CSV format not compatible. " + reason +
                 " Please upload a .csv file with a header row containing exactly these columns: "
                 + ", ".join(required_cols) + "."
    }), 400
  # No file in the request
  if "file" not in request.files:
    return csv_error("No file was received.")
  # File is not readable as a CSV
  try:
    input_data = pd.read_csv(request.files["file"])
  except Exception:
    return csv_error("The file could not be read as a CSV.")
  # Header only, no rows
  if input_data.empty:
    return csv_error("The file has no data rows.")
  # Columns must match exactly: nothing missing, nothing extra
  missing = [c for c in required_cols if c not in input_data.columns]
  extra = [c for c in input_data.columns if c not in required_cols]
  if missing or extra:
    reason = ""
    if missing:
      reason += "Missing columns: " + ", ".join(missing) + "."
    if extra:
      reason += " Unexpected columns (please remove): " + ", ".join(extra) + "."
    return csv_error(reason.strip())
  # Put the columns in training order (column order in the file does not matter)
  input_data = input_data[required_cols]
  # No blank cells allowed
  empty_cols = [c for c in required_cols if input_data[c].isnull().any()]
  if empty_cols:
    return csv_error("Empty values found in: " + ", ".join(empty_cols) + ".")
  # Run the model; any remaining problem with the values is reported as a CSV problem
  try:
    predictions = model.predict(input_data).tolist()
  except Exception:
    return csv_error("Some values could not be processed (check numbers are numeric and text values are spelled as expected).")
  # Map row index to predicted sales rounded to 2 decimal places
  return jsonify({str(i): round(p, 2) for i, p in enumerate(predictions)})


# Standard entry point to start the local Flask development server when executed directly
if __name__ == "__main__":
  superkart_api.run(host="0.0.0.0", port=7860)
