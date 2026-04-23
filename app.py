import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Datasets_AI_project/train_u6lujuX_CVtuZ9i.csv")
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Loan_Status_Y", axis=1)
y = data["Loan_Status_Y"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# UI Design
st.set_page_config(page_title="Explainable AI Loan Predictor", layout="centered")

st.title("💰 Loan Approval Predictor")
st.markdown("### Using Explainable AI Concepts")

st.write("Enter applicant details:")

# Inputs (simple sliders)
income = st.slider("Applicant Income", 0, 10000, 5000)
loan_amount = st.slider("Loan Amount", 0, 500, 150)
credit_history = st.selectbox("Credit History", [0, 1])

# Button
if st.button("Predict"):
    # Prepare input (basic mapping)
    input_data = np.zeros(len(X.columns))
    
    # Assign some important features
    if "ApplicantIncome" in X.columns:
        input_data[X.columns.get_loc("ApplicantIncome")] = income
    if "LoanAmount" in X.columns:
        input_data[X.columns.get_loc("LoanAmount")] = loan_amount
    if "Credit_History" in X.columns:
        input_data[X.columns.get_loc("Credit_History")] = credit_history

    prediction = model.predict([input_data])[0]

    result = "✅ Approved" if prediction == 1 else "❌ Not Approved"

    st.subheader("Result:")
    st.success(result)

    # Explainability (feature importance)
    st.subheader("Top Factors Influencing Decision")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(5)

    st.bar_chart(imp_df.set_index("Feature"))