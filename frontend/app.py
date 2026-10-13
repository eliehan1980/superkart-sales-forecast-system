# Import Streamlit for building the interactive web UI
import streamlit as st

# Import pandas for data manipulation and tabular transformations
import pandas as pd

# Import requests to send HTTP POST calls to the Flask REST API
import requests

# Base URL pointing to the deployed Flask backend microservice on Hugging Face Spaces
BACKEND_URL = "https://your-username-superkart-backend.hf.space"

# Render the primary header title for the dashboard
st.title("SuperKart Sales Forecast System")

# Display introductory text guiding the user on inputting data
st.write("Enter the product and store details below to predict the total sales.")

# Numeric input field for Product_Weight (default set to dataset median: 12.66 kg)
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)

# Dropdown selection for sugar category (default set to dataset mode: Low Sugar)
Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"],
    index=0
)

# Numeric input field for display area allocation ratio (default set to dataset median: 0.056)
Product_Allocated_Area = st.number_input("Product Allocated Area", min_value=0.0, value=0.056)

# Numeric input field for Maximum Retail Price (default set to dataset median: ₹146.74)
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=146.74)

# Dropdown selection for store physical footprint (default set to dataset mode: Medium)
Store_Size = st.selectbox(
    "Store Size",
    ["Medium", "High", "Small"],
    index=0
)

# Dropdown selection for geographic tier classification (default set to dataset mode: Tier 2)
Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 2", "Tier 1", "Tier 3"],
    index=0
)

# Dropdown selection for retail operating format (default set to dataset mode: Supermarket Type2)
Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"],
    index=0
)

# Dropdown selection for high-level SKU prefix (default set to dataset mode: FD)
Product_Id_char = st.selectbox(
    "Product ID Character",
    ["FD", "DR", "NC"],
    index=0
)

# Numeric input field for operational store age (default set to median store age: 16 years)
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, value=16)

# Dropdown selection for engineered shelf-life tier (default set to mode: Non Perishables)
Product_Type_Category = st.selectbox(
    "Product Type Category",
    ["Non Perishables", "Perishables"],
    index=0
)

# Assemble all user inputs into a dictionary matching the schema expected by the backend pipeline
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Render a prominent submission button to trigger single real-time prediction
if st.button("Predict", type='primary'):
    # Transmit the input dictionary as a JSON payload via an HTTP POST request to the Flask API
    response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data)

    # Check if the backend responded with HTTP 200 OK
    if response.status_code == 200:
        # Extract the forecasted sales value from the JSON response and display it formatted to 2 decimals
        st.success(f"Predicted Product Store Sales Total: ₹{response.json()['Sales']:.2f}")
    else:
        # Display an error banner if the API call fails or times out
        st.error("Unable to connect to the prediction API.")

# Render section header for processing multiple inventory records at once
st.subheader("Batch Prediction")

# File uploader widget accepting tabular CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Verify that a file has been uploaded before displaying the batch execution trigger
if uploaded_file is not None:
    # Render button to trigger batch prediction processing
    if st.button("Predict for Batch", type='primary'):
        # Send the raw uploaded file using multipart/form-data encoding to the batch endpoint
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file})

        # Check if the backend successfully processed the batch file
        if response.status_code == 200:
            # Display confirmation message indicating successful batch inference
            st.success("Predictions completed successfully!")

            # Convert JSON response dictionary into a transposed DataFrame and display it in an interactive table
            st.dataframe(pd.DataFrame([response.json()]).T.rename(columns={0: 'Predicted Sales'}))
        else:
            # Display an error message if the batch service request fails
            st.error("Unable to connect to the prediction API.")
