"""
explainer.py — SHAP-based Explainable AI Module
=================================================
Generates human-readable explanations for individual predictions
and overall feature importance using SHAP (SHapley Additive exPlanations).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Human-readable feature name mapping
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "age": "Age",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "thalch": "Maximum Heart Rate",
    "oldpeak": "ST Depression (Oldpeak)",
    "sex": "Gender (Male)",
    "fbs": "Fasting Blood Sugar > 120",
    "exang": "Exercise-Induced Angina",
    "cp_typical_angina": "Typical Angina Chest Pain",
    "cp_atypical_angina": "Atypical Angina Chest Pain",
    "cp_non_anginal": "Non-Anginal Chest Pain",
    "cp_asymptomatic": "Asymptomatic Chest Pain",
    "restecg_normal": "Normal ECG",
    "restecg_lv_hypertrophy": "LV Hypertrophy (ECG)",
    "restecg_st_t_abnormality": "ST-T Abnormality (ECG)",
}

# ---------------------------------------------------------------------------
# Human-readable explanation templates
# ---------------------------------------------------------------------------
EXPLANATION_TEMPLATES = {
    "age": {
        "positive": "Older age ({val}) increased risk",
        "negative": "Younger age ({val}) reduced risk",
    },
    "trestbps": {
        "positive": "High resting blood pressure ({val} mmHg) increased risk",
        "negative": "Normal resting blood pressure ({val} mmHg) reduced risk",
    },
    "chol": {
        "positive": "High cholesterol ({val} mg/dl) increased risk",
        "negative": "Normal cholesterol ({val} mg/dl) reduced risk",
    },
    "thalch": {
        "positive": "Lower maximum heart rate ({val} bpm) increased risk",
        "negative": "Higher maximum heart rate ({val} bpm) reduced risk",
    },
    "oldpeak": {
        "positive": "Significant ST depression ({val}) increased risk",
        "negative": "Minimal ST depression ({val}) reduced risk",
    },
    "sex": {
        "positive": "Being male increased risk",
        "negative": "Being female reduced risk",
    },
    "fbs": {
        "positive": "Elevated fasting blood sugar increased risk",
        "negative": "Normal fasting blood sugar reduced risk",
    },
    "exang": {
        "positive": "Exercise-induced angina increased risk",
        "negative": "No exercise-induced angina reduced risk",
    },
    "cp_asymptomatic": {
        "positive": "Asymptomatic chest pain type increased risk",
        "negative": "Non-asymptomatic chest pain reduced risk",
    },
    "cp_typical_angina": {
        "positive": "Typical angina chest pain increased risk",
        "negative": "Absence of typical angina reduced risk",
    },
    "cp_atypical_angina": {
        "positive": "Atypical angina chest pain increased risk",
        "negative": "Absence of atypical angina reduced risk",
    },
    "cp_non_anginal": {
        "positive": "Non-anginal chest pain pattern noted",
        "negative": "Non-anginal pattern absent — no significant effect",
    },
    "restecg_normal": {
        "positive": "Normal ECG pattern noted",
        "negative": "Normal ECG reduced risk",
    },
    "restecg_lv_hypertrophy": {
        "positive": "LV hypertrophy on ECG increased risk",
        "negative": "Absence of LV hypertrophy reduced risk",
    },
    "restecg_st_t_abnormality": {
        "positive": "ST-T abnormality on ECG increased risk",
        "negative": "Normal ST-T on ECG reduced risk",
    },
}


def _get_shap_values(model, X_input, feature_names):
    """
    Compute SHAP values using the model's coefficients (for Logistic Regression).
    For linear models, SHAP values ≈ coef * (x - mean), which is fast and exact.
    """
    # Use coefficient-based approximation for logistic regression
    coefficients = model.coef_[0]
    # SHAP value = coefficient * scaled_feature_value (already centered by scaler)
    shap_values = coefficients * X_input.flatten()
    return shap_values


def get_feature_importance(model, feature_names):
    """
    Return global feature importance from Logistic Regression coefficients.

    Returns
    -------
    list of dict : [{"feature": str, "importance": float}, ...]
        sorted by absolute importance descending.
    """
    coefficients = model.coef_[0]
    importance = []
    for fname, coef in zip(feature_names, coefficients):
        importance.append(
            {
                "feature": FEATURE_LABELS.get(fname, fname),
                "feature_key": fname,
                "importance": round(float(abs(coef)), 4),
                "direction": "increases risk" if coef > 0 else "decreases risk",
            }
        )
    importance.sort(key=lambda x: x["importance"], reverse=True)
    return importance


def explain_prediction(model, X_input, feature_names, raw_input=None):
    """
    Generate human-readable explanations for a single prediction.

    Parameters
    ----------
    model : trained LogisticRegression
    X_input : np.ndarray, shape (1, n_features) — scaled input
    feature_names : list of str
    raw_input : dict, optional — original unscaled values for display

    Returns
    -------
    explanations : list of str
        Top contributing factors in human-readable language.
    feature_contributions : dict
        {feature_name: shap_value} for all features.
    """
    shap_values = _get_shap_values(model, X_input, feature_names)

    # Build contributions dict
    contributions = {}
    for fname, sval in zip(feature_names, shap_values):
        label = FEATURE_LABELS.get(fname, fname)
        contributions[label] = round(float(sval), 4)

    # Generate human-readable explanations for top contributing features
    # Sort by absolute SHAP value
    indexed = list(enumerate(shap_values))
    indexed.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []
    for idx, sval in indexed[:6]:  # Top 6 factors
        fname = feature_names[idx]
        direction = "positive" if sval > 0 else "negative"

        if fname in EXPLANATION_TEMPLATES:
            template = EXPLANATION_TEMPLATES[fname][direction]
            # Try to inject raw value
            if raw_input and fname in raw_input:
                val = raw_input[fname]
            else:
                val = ""
            try:
                explanation = template.format(val=val)
            except (KeyError, IndexError):
                explanation = template.replace("{val}", str(val))
        else:
            label = FEATURE_LABELS.get(fname, fname)
            if sval > 0:
                explanation = f"{label} increased risk"
            else:
                explanation = f"{label} reduced risk"

        explanations.append(explanation)

    return explanations, contributions
