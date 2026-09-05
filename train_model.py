"""
Revive — AI Revenue Recovery Agent
Day 2: Train the recovery prediction model.

Run:
    python train_model.py

Produces: model.pkl (used by decision_engine.py)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("transactions.csv")

# IMPORTANT: drop the ground-truth probability column — using it would be leakage
df = df.drop(columns=["recovery_probability_true"], errors="ignore")

# --- Encode categorical features ---
categorical_cols = [
    "payment_method", "customer_type", "failure_reason",
    "merchant_category", "intervention"
]
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "amount", "attempt_count", "hours_since_failure",
    "previous_transactions", "previous_success_rate", "is_repeat_customer",
    "payment_method_enc", "customer_type_enc", "failure_reason_enc",
    "merchant_category_enc", "intervention_enc",
]

X = df[feature_cols]
y = df["recovered"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluate on held-out test set (this is what you report to judges) ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("=" * 50)
print("MODEL EVALUATION (held-out test set)")
print("=" * 50)
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print("=" * 50)

# --- Save model + encoders + feature list together ---
joblib.dump({
    "model": model,
    "encoders": encoders,
    "feature_cols": feature_cols,
}, "model.pkl")

print("\nSaved model.pkl")
