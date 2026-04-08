<<<<<<< HEAD
# Heart Disease Risk Prediction AI

An end-to-end Machine Learning web application predicting the probability of heart disease using a Logistic Regression model with Explainable AI (SHAP concepts) and simulated IoT Garment Integration.

## Features

- **Predictive Risk Modeling**: Predicts probability of heart disease (0 to 1).
- **Explainable AI (XAI)**: Generates human-readable explanations detailing why a specific prediction was made based on patient parameters.
- **Risk Classification & Recommendations**: Classifies the patient into Low, Medium, or High risk, providing tailored health guidance.
- **IoT Simulator**: Features a modular IoT framework allowing simulated wearable ECG input to be seamlessly injected into the prediction form.
- **Premium Glassmorphism UI**: Beautifully designed dark-themed front-end interface utilizing modern web design aesthetics.
- **Model Comparison**: Automatically trains and evaluates `Logistic Regression`, `Decision Tree`, and `Random Forest` models, outlining why `Logistic Regression` was selected (Interpretability & calibration).

## Project Structure

```
Heart Disease Risk Prediction/
├── backend/
│   └── app.py                  # Flask REST API
├── data/
│   └── heart_disease_uci.csv   # Source dataset
├── frontend/
│   ├── index.html              # Main web interface
│   ├── script.js               # Web app logic & API calls
│   └── style.css               # Styling and design system
├── model/
│   ├── train.py                # Model training, evaluation & comparison script
│   ├── heart_disease_model.pkl # Trained Logistic Regression model
│   ├── scaler.pkl              # Pre-trained StandardScaler
│   └── feature_names.pkl       # Feature column references
├── utils/
│   ├── __init__.py
│   ├── explainer.py            # SHAP-based feature explanations
│   ├── iot_simulator.py        # Simulated wearable input classes
│   ├── preprocessing.py        # Data cleaning and scaling tools
│   └── recommendations.py      # Health advice based on risk
├── requirements.txt            # Python dependencies
├── run.py                      # Master script (Train -> Start Server)
└── README.md
```

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
You can easily train the models and start the server with a single command:
```bash
python run.py
```
This script will:
1. Trigger the model training pipeline (`python model/train.py`).
2. Compare Logistic Regression with alternatives.
3. Save the best model (`.pkl`).
4. Start the Flask Server (`python backend/app.py`).

### 3. Open the Web Application
Navigate to `http://127.0.0.1:5000` in your web browser to use the tool.

## IoT Garment Integration

The application contains an expandable module (`utils/iot_simulator.py`) designed for wearable heart monitor data.
Currently, you can use the **Simulate ECG Data** button on the web application to automatically generate hardware input for features such as `Max Heart Rate`, `ST Depression (Oldpeak)`, `ECG Results`, and `Exercise Angina`.

## Disclaimer
*This tool is a prototype designed for educational purposes. It does not replace professional medical diagnosis.*
=======
# Heart_Disease_Risk_Prediction
>>>>>>> 235c4180fe9f1bb52355b66b14f8b45f7f620739
