"""
Revive — AI Revenue Recovery Agent
Decision Engine: given a failed transaction's features, predicts the
recovery probability under each possible action and recommends the one
with the highest expected financial value.

This module is imported by main.py — not run standalone (though the
__main__ block below lets you sanity-check it in isolation).
"""

import joblib
import numpy as np
import pandas as pd

_bundle = joblib.load("model.pkl")
_model = _bundle["model"]
_encoders = _bundle["encoders"]
_feature_cols = _bundle["feature_cols"]

# MVP scope: only two real options to compare
ACTIONS = ["do_nothing", "payment_link"]

# Cost of taking each action, as a fraction of transaction amount
ACTION_COST_FRACTION = {
    "do_nothing": 0.0,
    "payment_link": 0.0,  # no monetary cost, just the decision to send it
}


def _safe_encode(col, value):
    """Encode a categorical value; falls back to 0 if it's an unseen category."""
    le = _encoders[col]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0


def evaluate_transaction(txn: dict) -> dict:
    """
    txn should contain:
        amount, attempt_count, hours_since_failure, previous_transactions,
        previous_success_rate, is_repeat_customer, payment_method,
        customer_type, failure_reason, merchant_category

    Returns a dict with the recommendation and full expected-value breakdown.
    """
    results = {}

    for action in ACTIONS:
        row = {
            "amount": txn["amount"],
            "attempt_count": txn["attempt_count"],
            "hours_since_failure": txn["hours_since_failure"],
            "previous_transactions": txn["previous_transactions"],
            "previous_success_rate": txn["previous_success_rate"],
            "is_repeat_customer": txn["is_repeat_customer"],
            "payment_method_enc": _safe_encode("payment_method", txn["payment_method"]),
            "customer_type_enc": _safe_encode("customer_type", txn["customer_type"]),
            "failure_reason_enc": _safe_encode("failure_reason", txn["failure_reason"]),
            "merchant_category_enc": _safe_encode("merchant_category", txn["merchant_category"]),
            "intervention_enc": _safe_encode("intervention", action),
        }
        X = pd.DataFrame([row])[_feature_cols]
        prob = float(_model.predict_proba(X)[0][1])

        cost = ACTION_COST_FRACTION[action] * txn["amount"]
        expected_revenue = prob * txn["amount"]
        expected_profit = expected_revenue - cost

        results[action] = {
            "recovery_probability": round(prob, 4),
            "expected_revenue": round(expected_revenue, 2),
            "cost": round(cost, 2),
            "expected_profit": round(expected_profit, 2),
        }

    recommended_action = max(results, key=lambda a: results[a]["expected_profit"])

    return {
        "recommended_action": recommended_action,
        "options": results,
    }


if __name__ == "__main__":
    # Quick sanity check
    sample_txn = {
        "amount": 18000,
        "attempt_count": 1,
        "hours_since_failure": 2.0,
        "previous_transactions": 12,
        "previous_success_rate": 0.85,
        "is_repeat_customer": 1,
        "payment_method": "upi",
        "customer_type": "returning",
        "failure_reason": "temporary_failure",
        "merchant_category": "ecommerce",
    }
    result = evaluate_transaction(sample_txn)
    print(result)
