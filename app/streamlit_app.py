import streamlit as st
import pandas as pd
import requests

BENTOML_URL = "http://127.0.0.1:3001/predict"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
)

st.title("💳 Credit Card Transaction Fraud Detection")

st.markdown(
    """
This application evaluates credit card transactions using a machine learning model
to estimate fraud risk.

**Upload a CSV file containing PCA-transformed transaction data.**
"""
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Input Data Preview")
        st.write(df.head(20))

        st.markdown("---")

        if st.button("🚀 Run Fraud Prediction"):
            with st.spinner("Running model inference..."):
                response = requests.post(
                    BENTOML_URL,
                    json=df.to_dict(orient="records"),
                    timeout=300,
                )

            if response.status_code != 200:
                st.error("Failed to communicate with the model API.")
                st.stop()

            result = response.json()

            # Handle API error responses
            if result.get("status") == "error":
                st.error(result.get("message", "Unknown error from model API."))
                st.stop()

            results = result.get("results", [])

            if not results:
                st.warning("No prediction results returned.")
                st.stop()

            # Attach results to dataframe
            df_results = df.copy()
            df_results["fraud_probability"] = [
                r["fraud_probability"] for r in results
            ]
            df_results["risk_level"] = [
                r["risk_level"] for r in results
            ]
            df_results["decision"] = [
                r["decision"] for r in results
            ]

            st.subheader("✅ Prediction Results")
            st.write(df_results.head(50))

            st.success("Prediction completed successfully.")

            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Failed to process file: {e}")

st.markdown("---")

st.markdown(
    """
### ⚠️ Data Requirements

- This model **only supports PCA-transformed credit card transaction data**
- Required columns: `Time`, `V1`–`V28`, `Amount`
- Optional column: `Class` (ignored if present)
- Raw transaction data (merchant, location, card details) is **not supported**
- All values must be numeric and non-null

This tool is intended for **educational and experimental use only**.
"""
)
