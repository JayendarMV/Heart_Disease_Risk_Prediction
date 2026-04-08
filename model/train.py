"""
train.py — Model Training, Comparison & Persistence
=====================================================
Trains Logistic Regression, Decision Tree, and Random Forest on the
heart disease dataset. Evaluates and compares all models, then saves
the Logistic Regression model and preprocessing artefacts as .pkl files.
"""

import os
import sys
import pickle
import numpy as np

# Add project root to path so we can import utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from utils.preprocessing import preprocess_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "heart_disease_uci.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}


def main():
    print("=" * 60)
    print("  HEART DISEASE RISK PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Preprocess data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(DATA_PATH)
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # ------------------------------------------------------------------
    # 2. Train models
    # ------------------------------------------------------------------
    print("\n[2/5] Training models...")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, max_depth=5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(name, model, X_test, y_test)

    # ------------------------------------------------------------------
    # 3. Model comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("  " + "-" * 75)
    for name, metrics in results.items():
        print(
            f"  {name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}"
        )

    # ------------------------------------------------------------------
    # 4. Why Logistic Regression?
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  WHY LOGISTIC REGRESSION IS PREFERRED")
    print("=" * 60)
    print("""
  1. INTERPRETABILITY: Logistic Regression provides clear, interpretable
     coefficients — each feature's contribution to the prediction can be
     directly understood by clinicians and patients.

  2. PROBABILITY OUTPUT: Unlike Decision Trees, LR naturally outputs
     calibrated probabilities (0-1), essential for risk stratification
     (Low / Medium / High).

  3. CLINICAL TRUST: In healthcare, black-box models face regulatory and
     trust barriers. LR is widely accepted in medical literature and
     clinical decision support systems.

  4. EXPLAINABILITY: LR coefficients map directly to SHAP values, enabling
     human-readable explanations like "High cholesterol increased risk by X".

  5. COMPUTATIONAL EFFICIENCY: LR is lightweight, fast to train and predict,
     making it ideal for real-time IoT / wearable integrations.

  Note: While Random Forest may achieve slightly higher accuracy, the trade-off
  in interpretability and clinical applicability makes LR the better choice
  for this heart disease risk prediction system.
    """)

    # ------------------------------------------------------------------
    # 5. Save artefacts
    # ------------------------------------------------------------------
    print("[5/5] Saving model artefacts...")

    lr_model = models["Logistic Regression"]

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"  [OK] Model saved to: {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [OK] Scaler saved to: {SCALER_PATH}")

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"  [OK] Feature names saved to: {FEATURES_PATH}")

    # ------------------------------------------------------------------
    # Quick sanity check
    # ------------------------------------------------------------------
    print("\n[SANITY CHECK] Testing saved model...")
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)

    sample = X_test[0].reshape(1, -1)
    prob = loaded_model.predict_proba(sample)[0][1]
    print(f"  Sample probability: {prob:.4f}")
    print(f"  Prediction: {'Disease' if prob > 0.5 else 'No Disease'}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
