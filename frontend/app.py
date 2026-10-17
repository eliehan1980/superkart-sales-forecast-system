# Import Streamlit for building the interactive web UI
import streamlit as st
# Import pandas for data manipulation and tabular transformations
import pandas as pd
# Import requests to send HTTP POST calls to the Flask REST API
import requests

# Internal Docker network URL pointing directly to the backend container service on port 7860
BACKEND_URL = "http://sales-backend-instance:7860"

# The exact columns the backend expects in a batch file
REQUIRED_COLS = [
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
# Columns that must contain numbers only
NUMERIC_COLS = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
# Accepted text values for each categorical column
ALLOWED_VALUES = {
    "Product_Sugar_Content": ["Low Sugar", "Regular", "No Sugar"],
    "Store_Size": ["Small", "Medium", "High"],
    "Store_Location_City_Type": ["Tier 1", "Tier 2", "Tier 3"],
    "Store_Type": ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
    "Product_Id_char": ["FD", "DR", "NC"],
    "Product_Type_Category": ["Perishables", "Non Perishables"],
}


# Validate an uploaded batch file; returns blocking problems and non-blocking warnings
def check_csv(df):
    problems, warnings = [], []
    # Columns must match exactly: nothing missing, nothing extra (column order does not matter)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLS]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing) + ".")
    if extra:
        problems.append("Unexpected columns (please remove): " + ", ".join(extra) + ".")
    if df.empty:
        problems.append("The file has no data rows.")
    # Value checks only make sense once the structure is correct
    if problems:
        return problems, warnings
    # No blank cells allowed in required columns
    for col in REQUIRED_COLS:
        if df[col].isnull().any():
            problems.append("Empty values in '" + col + "'.")
    # Numeric columns must contain numbers only
    for col in NUMERIC_COLS:
        if not df[col].isnull().any() and pd.to_numeric(df[col], errors="coerce").isnull().any():
            problems.append("'" + col + "' must contain numbers only.")
    # Unrecognized category values are allowed, but the user is warned
    for col, allowed in ALLOWED_VALUES.items():
        bad = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if bad:
            warnings.append(
                "'" + col + "' has unrecognized value(s): " + ", ".join(bad)
                + ". Expected one of: " + ", ".join(allowed)
                + ". Predictions for these rows may be less accurate."
            )
    return problems, warnings


# Initialize session state so batch predictions persist across UI re-renders
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# Render the primary header title for the dashboard
st.title("SuperKart Sales Forecast System")

# Display introductory text guiding the user on inputting data
st.write("Enter the product and store details below to predict the total sales.")

# Numeric input field for Product_Weight (min: 4.0 kg, max: 25.0 kg, default: dataset median 12.66 kg)
Product_Weight = st.number_input("Product Weight", min_value=4.0, max_value=25.0, value=12.66)

# Dropdown selection for sugar category (default set to dataset mode: Low Sugar)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"], index=0)

# Numeric input field for display area allocation ratio (min: 0.001, max: 0.35, default: dataset median 0.056)
Product_Allocated_Area = st.number_input(
    "Product Allocated Area", min_value=0.001, max_value=0.35, value=0.056, format="%.3f"
)

# Numeric input field for Maximum Retail Price (min: 25.0, max: 300.0, default: dataset median ₹146.74)
Product_MRP = st.number_input("Product MRP", min_value=25.0, max_value=300.0, value=146.74)

# Dropdown selection for store physical footprint (default set to dataset mode: Medium)
Store_Size = st.selectbox("Store Size", ["Medium", "High", "Small"], index=0)

# Dropdown selection for geographic tier classification (default set to dataset mode: Tier 2)
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 2", "Tier 1", "Tier 3"], index=0)

# Dropdown selection for retail operating format (default set to dataset mode: Supermarket Type2)
Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type2", "Supermarket Type1", "Departmental Store", "Food Mart"],
    index=0,
)

# Dropdown selection for high-level SKU prefix (default set to dataset mode: FD)
Product_Id_char = st.selectbox("Product ID Character", ["FD", "DR", "NC"], index=0)

# Numeric input field for operational store age (min: 0, max: 50, default: median store age 16 years)
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=50, value=16)

# Dropdown selection for engineered shelf-life tier (default set to mode: Non Perishables)
Product_Type_Category = st.selectbox("Product Type Category", ["Non Perishables", "Perishables"], index=0)

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
    "Product_Type_Category": Product_Type_Category,
}

# Render a prominent submission button to trigger single real-time prediction
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=10)
        if response.status_code == 200:
            st.success(f"Predicted Product Store Sales Total: ₹{response.json()['Sales']:.2f}")
        else:
            st.error(f"Inference error ({response.status_code}): {response.text}")
    except requests.exceptions.Timeout:
        st.error("Request timed out after 10 seconds. Check backend performance.")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error connecting to backend: {e}")

# Render section header for processing multiple inventory records at once
st.subheader("Batch Prediction")

# Build a downloadable CSV template with the exact columns the backend expects, plus two example rows
template_df = pd.DataFrame(
    [
        {
            "Product_Weight": 12.66,
            "Product_Sugar_Content": "Low Sugar",
            "Product_Allocated_Area": 0.056,
            "Product_MRP": 146.74,
            "Store_Size": "Medium",
            "Store_Location_City_Type": "Tier 2",
            "Store_Type": "Supermarket Type2",
            "Product_Id_char": "FD",
            "Store_Age_Years": 16,
            "Product_Type_Category": "Non Perishables",
        },
        {
            "Product_Weight": 9.57,
            "Product_Sugar_Content": "No Sugar",
            "Product_Allocated_Area": 0.010,
            "Product_MRP": 123.67,
            "Store_Size": "Small",
            "Store_Location_City_Type": "Tier 3",
            "Store_Type": "Food Mart",
            "Product_Id_char": "NC",
            "Store_Age_Years": 27,
            "Product_Type_Category": "Non Perishables",
        },
    ]
)
# Offer the template as a one-click download
st.download_button(
    label="Download CSV template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="superkart_batch_template.csv",
    mime="text/csv",
)
# Show the accepted values so users know how to fill each column
st.caption(
    "Keep the header row unchanged and replace the example rows with your data. "
    "Product_Sugar_Content: Low Sugar, Regular, No Sugar | "
    "Store_Size: Small, Medium, High | "
    "Store_Location_City_Type: Tier 1, Tier 2, Tier 3 | "
    "Store_Type: Supermarket Type1, Supermarket Type2, Departmental Store, Food Mart | "
    "Product_Id_char: FD, DR, NC | "
    "Product_Type_Category: Perishables, Non Perishables."
)

# File uploader widget accepting tabular CSV files
uploaded_file = st.file_uploader("Upload CSV file for batch sales forecasting", type=["csv"])

if uploaded_file is not None:
    # Try to read the file; unreadable files get the same instruction as format problems
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception:
        batch_df = None
        st.error(
            "CSV format not compatible: the file could not be read. "
            "Please download the CSV template above and use it as the correct format."
        )
    if batch_df is not None:
        problems, warnings = check_csv(batch_df)
        if problems:
            # Block the prediction and tell the user how to fix the file
            st.error(
                "CSV format not compatible. " + " ".join(problems)
                + " Please download the CSV template above and use it as the correct format."
            )
        else:
            # Unrecognized category values are allowed, but the user is warned
            for w in warnings:
                st.warning(w)
            st.write("Uploaded Catalog Preview:", batch_df.head())
            if st.button("Predict Batch", type="secondary"):
                files = {"file": uploaded_file.getvalue()}
                try:
                    batch_response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files, timeout=30)
                    if batch_response.status_code == 200:
                        predictions = batch_response.json()
                        result_df = batch_df.copy()
                        result_df["Predicted_Sales"] = [predictions.get(str(i)) for i in range(len(result_df))]
                        st.session_state.batch_results = result_df
                        st.success("Batch forecasting complete!")
                    else:
                        st.error(f"Batch prediction failed ({batch_response.status_code}): {batch_response.text}")
                except requests.exceptions.Timeout:
                    st.error("Batch request timed out after 30 seconds.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Network error during batch transfer: {e}")

# Render persisted batch predictions with download utility
if st.session_state.batch_results is not None:
    st.dataframe(st.session_state.batch_results)
    csv_data = st.session_state.batch_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecasted Catalog (CSV)",
        data=csv_data,
        file_name="superkart_forecasted_catalog.csv",
        mime="text/csv",
    )
