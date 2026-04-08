"""
recommendations.py — Risk Classification & Health Recommendations
==================================================================
Classifies the predicted probability into risk levels and returns
personalized lifestyle / medical recommendations.
"""


def get_risk_level(probability: float) -> str:
    """
    Classify risk based on predicted probability.

    Parameters
    ----------
    probability : float (0 to 1)

    Returns
    -------
    str : "Low", "Medium", or "High"
    """
    if probability <= 0.3:
        return "Low"
    elif probability <= 0.7:
        return "Medium"
    else:
        return "High"


def get_recommendations(risk_level: str, raw_input: dict = None) -> list:
    """
    Return health recommendations based on risk level and patient data.

    Parameters
    ----------
    risk_level : str ("Low", "Medium", "High")
    raw_input : dict, optional — patient data for personalized advice

    Returns
    -------
    list of str : recommendations
    """
    recommendations = {
        "Low": [
            "✅ Maintain a healthy and balanced diet rich in fruits and vegetables",
            "✅ Continue regular physical activity (at least 30 min/day)",
            "✅ Keep stress levels in check with mindfulness or hobbies",
            "✅ Schedule routine health checkups annually",
            "✅ Stay hydrated and get adequate sleep (7-8 hours)",
        ],
        "Medium": [
            "⚠️ Monitor blood pressure and cholesterol levels regularly",
            "⚠️ Schedule checkups every 3-6 months",
            "⚠️ Reduce intake of saturated fats, salt, and processed foods",
            "⚠️ Increase physical activity — aim for 45 min of moderate exercise daily",
            "⚠️ Consider consulting a physician for a detailed cardiac assessment",
            "⚠️ Manage stress through yoga, meditation, or counseling",
            "⚠️ Limit alcohol consumption",
        ],
        "High": [
            "🚨 Consult a cardiologist immediately for a thorough evaluation",
            "🚨 Reduce cholesterol and sugar intake urgently",
            "🚨 Avoid smoking and limit alcohol consumption strictly",
            "🚨 Take prescribed medications regularly — do not skip doses",
            "🚨 Monitor blood pressure daily at home",
            "🚨 Follow a heart-healthy diet (DASH or Mediterranean diet)",
            "🚨 Engage only in doctor-approved physical activities",
            "🚨 Consider wearable heart monitoring for continuous tracking",
        ],
    }

    recs = recommendations.get(risk_level, recommendations["Medium"])

    # Add personalised tips based on raw input
    if raw_input:
        chol = raw_input.get("chol")
        trestbps = raw_input.get("trestbps")
        fbs = raw_input.get("fbs")
        thalch = raw_input.get("thalch")

        if chol and float(chol) > 240:
            recs.append("📌 Your cholesterol ({} mg/dl) is above the recommended limit of 200 mg/dl — dietary changes are advised".format(int(chol)))
        if trestbps and float(trestbps) > 140:
            recs.append("📌 Your resting BP ({} mmHg) is elevated — monitor frequently".format(int(trestbps)))
        if fbs and str(fbs).upper() in ("TRUE", "1", "YES"):
            recs.append("📌 Elevated fasting blood sugar detected — consider glucose management")
        if thalch and float(thalch) < 100:
            recs.append("📌 Your maximum heart rate ({} bpm) is low — discuss with your doctor".format(int(thalch)))

    return recs
