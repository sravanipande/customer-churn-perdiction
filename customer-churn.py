"""
Customer Churn Prediction - Business Insights Model
----------------------------------------------------
- Based on Telco Customer Churn dataset
- Performs feature engineering and model training
- Uses RandomForest and XGBoost for prediction
- Calculates expected revenue at risk for top customers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# Load Data 
def load_dataset(path="Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    return df

# Preprocessing 
def preprocess_data(df):
    df = df.copy()

    # Clean numeric columns
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))

    # Drop ID column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Encode target variable
    df['Churn_flag'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop('Churn', axis=1, inplace=True)

    # Tenure groups
    df['tenure_group'] = pd.cut(df['tenure'],
                                bins=[-1, 6, 12, 24, 48, 72, 1000],
                                labels=['0-6', '7-12', '13-24', '25-48', '49-72', '72+'])

    # Derived / business features
    df['MonthlyCharges_z'] = (df['MonthlyCharges'] - df['MonthlyCharges'].mean()) / df['MonthlyCharges'].std()
    df['High_Value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.85)).astype(int)
    df['approx_lifetime_value'] = df['tenure'] * df['MonthlyCharges']
    df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['contract_payment_combo'] = df['Contract'].astype(str) + "_" + df['PaymentMethod'].astype(str)

    # Convert Yes/No → 1/0 for remaining columns
    for col in df.select_dtypes(include='object').columns:
        if set(df[col].dropna().unique()) == {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    return df

#Build Pipelines
def build_models(df):
    X = df.drop('Churn_flag', axis=1)
    y = df['Churn_flag']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    # Handle imbalance for XGBoost
    scale_weight = (y.value_counts()[0] / y.value_counts()[1]) if len(y.value_counts()) == 2 else 1
    xgb_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42
    )

    rf_pipeline = Pipeline([('pre', preprocessor), ('clf', rf_model)])
    xgb_pipeline = Pipeline([('pre', preprocessor), ('clf', xgb_model)])

    return xgb_pipeline, rf_pipeline, X, y

# Training & Evaluation
def train_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Revenue Risk
    test_copy = X_test.copy()
    if 'tenure' in test_copy.columns and 'MonthlyCharges' in test_copy.columns:
        test_copy['approx_lifetime_value'] = test_copy['tenure'] * test_copy['MonthlyCharges']
        test_copy['churn_proba'] = proba
        test_copy['expected_revenue_at_risk'] = test_copy['approx_lifetime_value'] * test_copy['churn_proba']

        top_risk = test_copy.sort_values(by='expected_revenue_at_risk', ascending=False).head(10)
        print("\nTop 10 Customers by Revenue Risk:\n", top_risk[['tenure', 'MonthlyCharges', 'churn_proba', 'expected_revenue_at_risk']])
        top_risk.to_csv(f"top_{name}_revenue_risk.csv", index=False)

    joblib.dump(model, f"{name}.joblib")
    print(f"Saved model: {name}.joblib")

    return model

# SHAP Explainability
def shap_explain(model, X_train):
    try:
        pre = model.named_steps['pre']
        clf = model.named_steps['clf']
        Xt = pre.transform(X_train)
        explainer = shap.TreeExplainer(clf)
        sample = Xt[np.random.choice(Xt.shape[0], min(500, Xt.shape[0]), replace=False)]
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample)
    except Exception as e:
        print("SHAP explanation skipped:", e)

# Main Function 
def main():
    df = load_dataset()
    print("Dataset loaded:", df.shape)

    df = preprocess_data(df)
    print("After preprocessing:", df.shape)

    xgb_pipe, rf_pipe, X, y = build_models(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    rf_trained = train_evaluate(rf_pipe, X_train, X_test, y_train, y_test, "rf_churn")
    xgb_trained = train_evaluate(xgb_pipe, X_train, X_test, y_train, y_test, "xgb_churn")

    shap_explain(xgb_trained, X_train)

    print("\nTraining done. Models & reports saved successfully.")

if __name__ == "__main__":
    main()
