# 💳 Credit Card Fraud Detection - Interactive Portfolio App

This project is a Streamlit-based web application for detecting fraudulent credit card transactions using a trained machine learning model.

## Features

- Upload your own transaction CSV file for fraud prediction
- Manual entry for single transaction prediction
- Highlights detected fraudulent transactions
- Interactive data preview and results

## Files

- `app.py`: Main Streamlit app
- `creditcard.csv`: Example dataset
- `fraud_model.pkl`: Trained ML pipeline (Random Forest + scaler)
- `FRAUDDETECTION.ipynb`: Model training and analysis notebook
- `requirements.txt`: Python dependencies

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```sh
   streamlit run app.py
   ```

3. **Upload a CSV or enter transaction details manually.**

## Model

- Trained using Random Forest with data preprocessing (scaling, imbalance handling).
- Saved as a pipeline in `fraud_model.pkl`.

## Requirements

See [`requirements.txt`](requirements.txt).

## Live at : https://fraud-detection-api-ugabyqm5yucb2sxijtymt8.streamlit.app/


