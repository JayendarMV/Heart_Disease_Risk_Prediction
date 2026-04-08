"""
preprocessing.py — Data Cleaning, Encoding, Scaling & Pipeline
================================================================
Handles all data transformation from raw CSV to model-ready features.
Supports both batch (training) and single-input (inference) modes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Features used by the model (after preprocessing / one-hot encoding)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = {
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "restecg": ["normal", "lv hypertrophy", "st-t abnormality"],
}

# Default / median values used when a field is left blank (optional fields)
DEFAULTS = {
    "trestbps": 130,
    "chol": 240,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150,
    "exang": False,
    "oldpeak": 0.0,
}

# Mandatory fields — the prediction won't proceed without these
MANDATORY_FIELDS = ["age", "sex", "cp"]


# ---------------------------------------------------------------------------
# Batch preprocessing (for training)
# ---------------------------------------------------------------------------
def preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load the raw CSV, clean, encode, scale, and split.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
    scaler : fitted StandardScaler
    feature_names : list of column names after encoding
    """
    df = pd.read_csv(csv_path)

    # ---- Drop unused columns ----
    drop_cols = ["id", "dataset", "slope", "ca", "thal"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ---- Binarise target (num): 0 = no disease, 1+ = disease ----
    df["target"] = (df["num"].astype(float) >= 1).astype(int)
    df.drop(columns=["num"], inplace=True)

    # ---- Handle missing / invalid numeric values ----
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Cholesterol == 0 is physiologically impossible → treat as missing
            if col == "chol":
                df.loc[df[col] == 0, col] = np.nan
            df[col].fillna(df[col].median(), inplace=True)

    # ---- Handle missing binary values ----
    for col in BINARY_FEATURES:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(
                    {"Male": 1, "Female": 0, "TRUE": 1, "FALSE": 0, True: 1, False: 0}
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = df[col].astype(int)

    # ---- One-hot encode categoricals ----
    for col, categories in CATEGORICAL_FEATURES.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col].replace({"nan": categories[0], "": categories[0]}, inplace=True)
            for cat in categories:
                df[f"{col}_{cat.replace(' ', '_').replace('-', '_')}"] = (
                    df[col] == cat
                ).astype(int)
            df.drop(columns=[col], inplace=True)

    # ---- Separate features / target ----
    y = df["target"].values
    X = df.drop(columns=["target"])
    feature_names = list(X.columns)

    # ---- Scale ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ---------------------------------------------------------------------------
# Single-input preprocessing (for inference / API)
# ---------------------------------------------------------------------------
def preprocess_single_input(input_dict: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Convert a single user-input dictionary into a scaled feature vector
    aligned with the training feature order.

    Parameters
    ----------
    input_dict : dict
        Raw user input (may have missing optional fields).
    scaler : StandardScaler
        Fitted scaler from training.
    feature_names : list
        Column order from training.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    # Fill defaults for missing optional fields
    data = {}
    for key, default in DEFAULTS.items():
        data[key] = input_dict.get(key, default)
        if data[key] is None or data[key] == "":
            data[key] = default

    # Mandatory fields (no defaults)
    data["age"] = float(input_dict["age"])
    data["sex"] = input_dict.get("sex", "Male")
    data["cp"] = input_dict.get("cp", "asymptomatic")

    # Build feature vector following same encoding as training
    row = {}

    # Numeric
    for col in NUMERIC_FEATURES:
        row[col] = float(data.get(col, DEFAULTS.get(col, 0)))

    # Binary
    sex_val = data["sex"]
    if isinstance(sex_val, str):
        row["sex"] = 1 if sex_val.strip().lower() == "male" else 0
    else:
        row["sex"] = int(sex_val)

    fbs_val = data["fbs"]
    if isinstance(fbs_val, str):
        row["fbs"] = 1 if fbs_val.strip().upper() == "TRUE" else 0
    elif isinstance(fbs_val, bool):
        row["fbs"] = int(fbs_val)
    else:
        row["fbs"] = int(fbs_val)

    exang_val = data["exang"]
    if isinstance(exang_val, str):
        row["exang"] = 1 if exang_val.strip().upper() == "TRUE" else 0
    elif isinstance(exang_val, bool):
        row["exang"] = int(exang_val)
    else:
        row["exang"] = int(exang_val)

    # One-hot categoricals
    cp_val = str(data["cp"]).strip().lower()
    for cat in CATEGORICAL_FEATURES["cp"]:
        col_name = f"cp_{cat.replace(' ', '_').replace('-', '_')}"
        row[col_name] = 1 if cp_val == cat else 0

    restecg_val = str(data.get("restecg", "normal")).strip().lower()
    for cat in CATEGORICAL_FEATURES["restecg"]:
        col_name = f"restecg_{cat.replace(' ', '_').replace('-', '_')}"
        row[col_name] = 1 if restecg_val == cat else 0

    # Align with training feature order
    vector = [row.get(f, 0) for f in feature_names]
    vector = np.array(vector).reshape(1, -1)

    # Scale
    vector_scaled = scaler.transform(vector)
    return vector_scaled
