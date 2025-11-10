# customer-churn-prediction
Machine learning project that predicts customer churn using Random Forest and XGBoost classifiers. Includes business insights like expected revenue at risk, feature explainability (SHAP), and an interactive Streamlit dashboard.

This project predicts which customers are most likely to discontinue their service (churn) using historical customer data.
It combines machine learning with business insights to help companies reduce revenue loss and improve retention.

Features

Interactive Streamlit Dashboard – Easy-to-use web app for exploring churn risk.
Machine Learning Models – Built with Random Forest and XGBoost for high accuracy.
Customer Insights – Highlights high-value customers who are at risk.
Feature Importance (SHAP) – Explains which features contribute most to churn.
Downloadable Results – Export predictions and risk analysis reports as CSV files.

How It Works

Data Loading
Upload a customer dataset (e.g., the Telco Customer Churn dataset).

Preprocessing
The app cleans data, handles missing values, and encodes categorical variables.

Feature Engineering
approx_lifetime_value: Estimates the customer’s long-term worth.
expected_revenue_at_risk: Calculates potential revenue loss if the customer churns.

Model Training
Trains Random Forest and XGBoost models using an 80-20 split.
Displays accuracy and confusion matrix.

Prediction & Insights
Predicts churn probability for each customer.
Displays top 10 high-risk, high-value customers.
Uses SHAP plots to explain which features most influence predictions.
