import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

#Load Models 
@st.cache_resource
def load_models():
    rf = joblib.load("rf_churn.joblib")
    xgb = joblib.load("xgb_churn.joblib")
    return rf, xgb

rf_model, xgb_model = load_models()

#Title & Intro
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This tool predicts **customer churn** using trained Random Forest and XGBoost models,  
and estimates **expected revenue loss** for high-risk customers.
""")

st.sidebar.header("Enter Customer Information")

# User Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

#  Create DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior == "Yes" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "MultipleLines": [1 if multiple_lines == "Yes" else 0],
    "InternetService": [internet_service],
    "OnlineSecurity": [1 if online_security == "Yes" else 0],
    "OnlineBackup": [1 if online_backup == "Yes" else 0],
    "DeviceProtection": [1 if device_protection == "Yes" else 0],
    "TechSupport": [1 if tech_support == "Yes" else 0],
    "StreamingTV": [1 if streaming_tv == "Yes" else 0],
    "StreamingMovies": [1 if streaming_movies == "Yes" else 0],
    "Contract": [contract],
    "PaperlessBilling": [1 if paperless_billing == "Yes" else 0],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Extra Features 
input_data["MonthlyCharges_z"] = (
    (input_data["MonthlyCharges"] - input_data["MonthlyCharges"].mean()) /
    input_data["MonthlyCharges"].std()
)
input_data["High_Value"] = (input_data["MonthlyCharges"] >
                            input_data["MonthlyCharges"].quantile(0.85)).astype(int)
input_data["approx_lifetime_value"] = input_data["tenure"] * input_data["MonthlyCharges"]
input_data["avg_monthly_spend"] = input_data["TotalCharges"] / (input_data["tenure"] + 1)
input_data["tenure_group"] = pd.cut(
    input_data["tenure"],
    bins=[-1, 6, 12, 24, 48, 72, 1000],
    labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"]
)
input_data["contract_payment_combo"] = (
    input_data["Contract"].astype(str) + "_" + input_data["PaymentMethod"].astype(str)
)

# Prediction 
model_choice = st.radio("Select Model", ["Random Forest", "XGBoost"])

if st.button("Predict Churn"):
    model = rf_model if model_choice == "Random Forest" else xgb_model

    # Check and fill missing columns
    expected_cols = (
        model.named_steps['pre'].transformers_[0][2] +
        model.named_steps['pre'].transformers_[1][2]
    )
    for c in expected_cols:
        if c not in input_data.columns:
            input_data[c] = np.nan

    churn_prob = model.predict_proba(input_data)[0][1]
    churn_result = model.predict(input_data)[0]

    st.subheader("Prediction Results")
    st.write(f"**Churn Probability:** {churn_prob * 100:.2f}%")
    st.write(f"**Prediction:** {'Likely to Churn' if churn_result == 1 else 'Will Stay'}")

    expected_loss = churn_prob * input_data["approx_lifetime_value"][0]
    st.write(f"**Expected Revenue at Risk:**  ${expected_loss:.2f}")

    #  SHAP Explainability 
    with st.expander("Model Explainability (SHAP)"):
        try:
            clf = model.named_steps["clf"]
            pre = model.named_steps["pre"]
            processed = pre.transform(input_data)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(processed)
            shap.initjs()
            st.write("Feature impact for this prediction:")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, processed, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP not available: {e}")

# Top Risk Customers 
st.markdown("---")
st.header("Top 10 Customers by Expected Revenue Risk")

try:
    top_risk = pd.read_csv("top_xgb_revenue_risk.csv")
    st.dataframe(top_risk.head(10))
except FileNotFoundError:
    st.info("Please run the main training script to generate 'top_xgb_revenue_risk.csv'.")
