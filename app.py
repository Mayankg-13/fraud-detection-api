import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Load model
model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection - Interactive Portfolio App")

st.markdown("""
### 📌 Project Overview
Credit card fraud is a critical issue worldwide.  

This app demonstrates a **Machine Learning model** that detects fraudulent transactions.  

**Steps in project:**
- Data preprocessing (handling imbalance, scaling, feature selection)
- Model training (Random Forest)
- Evaluation (Accuracy, Precision, Recall, F1-Score)
""")

# -------------------------------
# 📂 Upload CSV file
# -------------------------------
st.header("🔎 Try Fraud Detection on Your Data")
uploaded_file = st.file_uploader("Upload transaction CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📂 Uploaded Data Preview:", data.head())

    # ✅ Drop target column if present
    if "Class" in data.columns:
        data_features = data.drop("Class", axis=1)
    else:
        data_features = data

    # ✅ Predictions
    preds = model.predict(data_features)
    data["Fraud_Prediction"] = preds
    st.write("✅ Predictions:", data.head())

    # ✅ Highlight fraud
    frauds = data[data["Fraud_Prediction"] == 1]
    st.write("⚠️ Detected Fraudulent Transactions:", frauds)

# -------------------------------
# ✍️ Manual Input Section
# -------------------------------
st.subheader("Or Enter Transaction Manually")

amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
time = st.number_input("Transaction Time", min_value=0.0, format="%.2f")

if st.button("Predict"):
    # ✅ Create full feature list used during training
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # ✅ Fill all features with 0 except Amount & Time
    input_data = [0] * len(feature_names)
    input_data[0] = time                  # Time
    input_data[-1] = amount               # Amount

    # ✅ Convert to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # ✅ Prediction
    pred = model.predict(input_df)[0]
    st.write("🔮 Prediction:", "Fraud ❌" if pred == 1 else "Legit ✅")
