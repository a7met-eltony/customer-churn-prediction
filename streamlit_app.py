import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow import keras

import churn_features  # noqa: F401  (needed to load the saved pipelines)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_comparison():
    return pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

@st.cache_resource
def load_models():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    selection = joblib.load(MODELS_DIR / "model_selection.joblib")
    ml_models = {}
    for file in MODELS_DIR.glob("*.joblib"):
        if file.name not in {"preprocessor.joblib", "model_selection.joblib"}:
            ml_models[file.stem] = joblib.load(file)
    dl_model = keras.models.load_model(MODELS_DIR / "deep_learning_model.keras")
    return preprocessor, selection, ml_models, dl_model

df = load_data()
comparison = load_comparison()
try:
    preprocessor, selection, ml_models, dl_model = load_models()
    artifacts_ready = True
except Exception as exc:
    artifacts_ready = False
    preprocessor = selection = dl_model = None
    ml_models = {}

st.title("Customer Churn Prediction")
st.caption("Machine Learning + Deep Learning system for predicting customer churn.")

if not artifacts_ready:
    st.warning(
        "Model artifacts are not generated yet. Run all cells in `project.ipynb` once; "
        "the notebook will create the models/ and outputs/ files required by this app."
    )
    st.stop()

best_model = selection["best_model"]
best_f1 = selection["best_f1"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("Churn Rate", f"{df['Churn'].eq('Yes').mean():.1%}")
m3.metric("Best Model", best_model)
m4.metric("Best F1-Score", f"{best_f1:.3f}")

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "EDA"])

with tab1:
    st.subheader("Predict Churn for a New Customer")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
        total = st.number_input("Total Charges", min_value=0.0, value=float(monthly * max(tenure, 1)), step=10.0)
        model_choice = st.selectbox(
            "Model",
            ["Best Model"] + list(comparison["Model"])
        )

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    selected_model = best_model if model_choice == "Best Model" else model_choice

    if st.button("Predict Churn", type="primary", use_container_width=True):
        if selected_model == "Deep Learning":
            x_new = preprocessor.transform(customer).astype("float32")
            probability = float(dl_model.predict(x_new, verbose=0).ravel()[0])
        else:
            model_key = selected_model.lower().replace(" ", "_")
            probability = float(ml_models[model_key].predict_proba(customer)[:, 1][0])

        prediction = "Churn" if probability >= 0.50 else "No Churn"

        if prediction == "Churn":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.metric("Churn Probability", f"{probability:.2%}")
        st.progress(probability)
        st.dataframe(customer, use_container_width=True)

with tab2:
    st.subheader("Model Performance")
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(
        comparison[display_cols].style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
        kind="bar", ylim=(0, 1), ax=ax
    )
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.image(str(OUTPUT_DIR / "confusion_matrices.png"), caption="Confusion Matrices")
    st.image(str(OUTPUT_DIR / "dl_training_curves.png"), caption="Deep Learning Training Curves")

with tab3:
    st.subheader("Exploratory Data Analysis")
    st.image(str(OUTPUT_DIR / "eda_churn_distribution.png"), caption="Churn Distribution")
    st.image(str(OUTPUT_DIR / "eda_tenure_vs_churn.png"), caption="Tenure vs Churn")
    st.image(str(OUTPUT_DIR / "eda_monthly_charges_vs_churn.png"), caption="Monthly Charges vs Churn")
    st.image(str(OUTPUT_DIR / "eda_categorical_churn_rates.png"), caption="Categorical Churn Rates")
    st.image(str(OUTPUT_DIR / "eda_correlation_heatmap.png"), caption="Correlation Heatmap")

    # Feature engineering / selection / balancing plots (created by the notebook).
    extra_plots = [
        ("eda_engineered_features.png", "Engineered Features vs Churn"),
        ("eda_feature_importance_mi.png", "Feature Selection (Mutual Information)"),
        ("eda_class_balance_before_after.png", "Class Balance Before vs After SMOTE"),
    ]
    for file_name, caption in extra_plots:
        if (OUTPUT_DIR / file_name).exists():
            st.image(str(OUTPUT_DIR / file_name), caption=caption)

st.divider()
st.caption("Project: AI Customer Churn Prediction | ML + Deep Learning")
