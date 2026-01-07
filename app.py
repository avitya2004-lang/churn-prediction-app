import streamlit as st

# SET PAGE CONFIG HARUS PALING ATAS!
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

import pandas as pd
import numpy as np
import joblib
import json

# Load model dan scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_churn_model_v2.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    return model, scaler, model_columns

model, scaler, model_columns = load_model()

# Judul aplikasi
st.title("📊 Customer Churn Prediction")
st.markdown("Prediksi apakah pelanggan akan **churn** (berhenti berlangganan) atau tidak")
st.markdown("---")

# Sidebar untuk input
st.sidebar.header("📋 Input Data Pelanggan")

# Informasi demografis
st.sidebar.subheader("👤 Informasi Demografis")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# Layanan
st.sidebar.subheader("📱 Layanan yang Digunakan")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# Informasi akun
st.sidebar.subheader("💳 Informasi Akun")
tenure = st.sidebar.slider("Tenure (bulan)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, 0.1)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, 1.0)

# Tombol prediksi
predict_button = st.sidebar.button("🔮 Prediksi Churn", use_container_width=True)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Data Input")
    input_data = {
        "Gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    df_input = pd.DataFrame([input_data])
    st.dataframe(df_input.T, use_container_width=True)

with col2:
    st.subheader("🎯 Hasil Prediksi")
    
    if predict_button:
        with st.spinner('Melakukan prediksi...'):
            try:
                # Preprocessing
                df_processed = df_input.copy()
                
                # Mapping nama kolom
                column_mapping = {
                    "Gender": "gender",
                    "Partner": "Partner",
                    "Dependents": "Dependents",
                    "Tenure": "tenure",
                    "PhoneService": "PhoneService",
                    "MultipleLines": "MultipleLines",
                    "InternetService": "InternetService",
                    "OnlineSecurity": "OnlineSecurity",
                    "OnlineBackup": "OnlineBackup",
                    "DeviceProtection": "DeviceProtection",
                    "TechSupport": "TechSupport",
                    "StreamingTV": "StreamingTV",
                    "StreamingMovies": "StreamingMovies",
                    "Contract": "Contract",
                    "PaperlessBilling": "PaperlessBilling",
                    "PaymentMethod": "PaymentMethod",
                    "MonthlyCharges": "MonthlyCharges",
                    "TotalCharges": "TotalCharges"
                }
                
                df_processed.rename(columns=column_mapping, inplace=True)
                
                # One-Hot Encoding
                df_encoded = pd.get_dummies(df_processed, drop_first=True)
                
                # Align kolom dengan model
                for col in model_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                
                df_encoded = df_encoded[model_columns]
                
                # Scaling
                df_scaled = scaler.transform(df_encoded)
                
                # Prediksi
                prediction = model.predict(df_scaled)[0]
                prediction_proba = model.predict_proba(df_scaled)[0]
                
                # Tampilkan hasil
                if prediction == "Yes":
                    st.error("⚠️ **CHURN PREDICTION: YES**")
                    st.markdown(f"Pelanggan ini **berpotensi churn** dengan probabilitas **{prediction_proba[1]*100:.2f}%**")
                    st.markdown("### 💡 Rekomendasi:")
                    st.markdown("- Hubungi pelanggan untuk memberikan penawaran khusus")
                    st.markdown("- Tawarkan diskon atau upgrade layanan")
                    st.markdown("- Tanyakan feedback untuk meningkatkan kepuasan")
                else:
                    st.success("✅ **CHURN PREDICTION: NO**")
                    st.markdown(f"Pelanggan ini **kemungkinan tetap berlangganan** dengan probabilitas **{prediction_proba[0]*100:.2f}%**")
                    st.markdown("### 💡 Rekomendasi:")
                    st.markdown("- Pertahankan kualitas layanan")
                    st.markdown("- Berikan reward untuk loyalitas")
                    st.markdown("- Tawarkan program referral")
                
                # Progress bar probabilitas
                st.markdown("### 📊 Probabilitas")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("No Churn", f"{prediction_proba[0]*100:.2f}%")
                with col_b:
                    st.metric("Churn", f"{prediction_proba[1]*100:.2f}%")
                
                st.progress(prediction_proba[1])
                
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {str(e)}")
                st.info("Silakan coba lagi atau hubungi admin")

# Footer
st.markdown("---")
st.markdown("### 📋 Informasi Model")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Model", "Logistic Regression")
with col_info2:
    st.metric("Accuracy", "80.70%")
with col_info3:
    st.metric("Recall", "56.68%")

st.markdown("**Dibuat oleh:** Avitya | **NIM:** A11.2022.14783")
