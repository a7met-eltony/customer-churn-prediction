"""Feature engineering shared by the notebook and the Streamlit app.

Keeping this in its own module (instead of inside the notebook) lets joblib
load the saved pipelines later, because the pickled pipeline refers to
`churn_features.add_features`.
"""
import numpy as np
import pandas as pd

ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


def add_features(X):
    """Take the raw customer columns and add engineered columns (no target used)."""
    X = X.copy()

    # Make sure numeric columns are numeric (works for raw CSV rows and app input).
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 1) How many add-on services the customer actually uses.
    X["num_addon_services"] = sum((X[c] == "Yes").astype(int) for c in ADDON_SERVICES)

    # 2) Protection / support services (churn is much higher without them).
    X["num_protection_services"] = sum((X[c] == "Yes").astype(int) for c in PROTECTION_SERVICES)

    # 3) Average monthly spend over the customer's lifetime (+1 avoids division by zero).
    X["avg_monthly_spend"] = X["TotalCharges"] / (X["tenure"] + 1)

    # 4) Is the customer paying more now than their historical average?
    X["charge_vs_avg_ratio"] = X["MonthlyCharges"] / (X["avg_monthly_spend"] + 1e-6)

    # 5) Tenure bucket: new customers churn far more than loyal ones.
    X["tenure_group"] = pd.cut(
        X["tenure"], bins=[-1, 12, 24, 48, 72, np.inf],
        labels=["0-12", "13-24", "25-48", "49-72", "72+"],
    ).astype(str)

    # 6) Automatic payment (credit card / bank transfer) vs manual payment.
    X["is_autopay"] = X["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

    # 7) Has internet service at all.
    X["has_internet"] = (X["InternetService"] != "No").astype(int)

    return X
