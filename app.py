import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Bank Customer Churn Prediction")
st.markdown("Upload customer data to predict churn risk.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the data
    input_df = pd.read_csv(uploaded_file)

    # Drop target and non-feature columns
    columns_to_drop = ['Exited', 'RowNumber', 'CustomerId', 'Surname']
    input_df = input_df.drop(columns=[col for col in columns_to_drop if col in input_df.columns])

    # One-hot encode categorical columns
    input_df = pd.get_dummies(input_df, columns=['Geography', 'Gender'], drop_first=True)

    # Ensure the order and presence of features as used during training
    expected_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_Germany', 'Geography_Spain', 'Gender_Male'
    ]

    # Add missing columns if any (from one-hot encoding)
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_features]

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make predictions
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[:, 1]

    # Combine with results
    results_df = pd.DataFrame({
        "Churn Prediction": prediction,
        "Churn Probability": probability
    })

    st.write("### Prediction Results")
    st.write(results_df)

    # Download results
    st.download_button(
        label="Download Prediction Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="churn_prediction_results.csv",
        mime="text/csv"
    )
