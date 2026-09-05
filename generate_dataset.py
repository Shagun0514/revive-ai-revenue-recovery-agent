"""
Merchant Revenue Autopilot — Synthetic Dataset Generator
Track 03: AI Revenue Recovery
"""

import argparse
import numpy as np
import pandas as pd


PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet", "emi"]
FAILURE_REASONS = [
    "temporary_failure",
    "insufficient_funds",
    "bank_declined",
    "network_error",
    "otp_timeout",
    "gateway_timeout",
]
MERCHANT_CATEGORIES = ["ecommerce", "subscription", "travel", "food_delivery", "education"]

ACTIONS = ["do_nothing", "payment_link"]

ACTION_COST_FRACTION = {
    "do_nothing": 0.0,
    "payment_link": 0.0,
}

ACTION_BASE_LIFT = {
    "do_nothing": 0.00,
    "payment_link": 0.30,
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_type = rng.choice(["new", "returning"], size=rows, p=[0.4, 0.6])
    total_prior_transactions = np.where(
        customer_type == "new",
        rng.integers(0, 2, size=rows),
        rng.integers(2, 40, size=rows),
    )
    prior_success_rate = np.where(
        customer_type == "new",
        rng.uniform(0.0, 0.5, size=rows),
        rng.beta(6, 2, size=rows),
    )

    amount = np.round(rng.lognormal(mean=8.5, sigma=0.9, size=rows), -1)
    amount = np.clip(amount, 100, 150000)

    payment_method = rng.choice(PAYMENT_METHODS, size=rows, p=[0.45, 0.25, 0.15, 0.1, 0.05])
    failure_reason = rng.choice(
        FAILURE_REASONS, size=rows,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    )
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=rows)

    attempt_count = rng.poisson(lam=1.4, size=rows) + 1
    attempt_count = np.clip(attempt_count, 1, 6)

    time_since_failure_hrs = rng.exponential(scale=6, size=rows)

    action = rng.choice(ACTIONS, size=rows, p=[0.45, 0.55])

    z = 0.0
    z += 1.8 * prior_success_rate
    z += 0.35 * (customer_type == "returning").astype(float)
    z -= 0.15 * np.log1p(amount / 1000)
    z -= 0.12 * (attempt_count - 1)
    z -= 0.10 * np.log1p(time_since_failure_hrs)
    z -= np.where(failure_reason == "insufficient_funds", 0.5, 0.0)
    z -= np.where(failure_reason == "bank_declined", 0.35, 0.0)
    z += np.where(failure_reason == "temporary_failure", 0.25, 0.0)
    z += np.where(failure_reason == "otp_timeout", 0.15, 0.0)
    action_lift = np.array([ACTION_BASE_LIFT[a] for a in action])
    z += action_lift * 3.0
    z += rng.normal(0, 0.35, size=rows)

    recovery_prob = sigmoid(z - 1.2)
    recovered = rng.binomial(1, recovery_prob)

    intervention_cost = np.round(
        np.array([ACTION_COST_FRACTION[a] for a in action]) * amount, 2
    )
    recovered_amount = np.where(recovered == 1, amount, 0.0)

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:06d}" for i in range(rows)],
        "customer_id": [f"cust_{rng.integers(0, rows // 2):05d}" for _ in range(rows)],
        "amount": amount,
        "payment_method": payment_method,
        "customer_type": customer_type,
        "previous_transactions": total_prior_transactions,
        "previous_success_rate": np.round(prior_success_rate, 3),
        "failure_reason": failure_reason,
        "attempt_count": attempt_count,
        "hours_since_failure": np.round(time_since_failure_hrs, 2),
        "is_repeat_customer": (customer_type == "returning").astype(int),
        "merchant_category": merchant_category,
        "intervention": action,
        "intervention_cost": intervention_cost,
        "recovery_probability_true": np.round(recovery_prob, 4),
        "recovered": recovered,
        "recovered_amount": recovered_amount,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic revenue-recovery dataset")
    parser.add_argument("--rows", type=int, default=2500, help="Number of transactions to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default="transactions.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows -> {args.out}")
    print(f"Overall recovery rate: {df['recovered'].mean():.1%}")
    print("\nRecovery rate by action:")
    print(df.groupby("intervention")["recovered"].mean().sort_values(ascending=False).round(3))
    print("\nRecovery rate by customer_type:")
    print(df.groupby("customer_type")["recovered"].mean().round(3))
    print(f"\nTotal transaction value: ₹{df['amount'].sum():,.0f}")
    print(f"Total recovered value:   ₹{df['recovered_amount'].sum():,.0f}")


if __name__ == "__main__":
    main()