# Revive — AI Revenue Recovery Agent

**Track 03: AI Revenue Recovery**

> Revive treats every failed payment as a financial decision, not just a database entry. It predicts whether recovery is worth pursuing, decides the most profitable action, checks it's allowed to act, executes through real Razorpay infrastructure, and proves the money actually came back.

---

## The Problem

Merchants lose revenue even when customers want to complete a purchase — a payment fails from a temporary bank issue, a network blip, an expired session. Most merchants have no system that automatically decides *whether it's worth recovering* and *acts on it*. Revive closes that loop end to end.

## What It Does

1. **Detects** — a Razorpay webhook fires the instant a payment fails
2. **Predicts** — an XGBoost model trained on transaction history estimates recovery probability under two options: do nothing, or send a payment link
3. **Decides** — compares expected profit (`probability × amount`) across both options and picks the higher one
4. **Explains** — an LLM (Claude Haiku) turns the decision into a plain-language explanation for the merchant
5. **Checks policy** — amounts under ₹10,000 auto-execute; amounts at or above that threshold are held for human approval instead of acted on unilaterally
6. **Acts** — creates a real Razorpay payment link via their API (Test Mode)
7. **Confirms** — a second webhook (`payment_link.paid`) confirms the customer actually paid, and the recovered amount is logged
8. **Shows it** — a live dashboard displays revenue at risk, revenue recovered, recovery rate, and a full decision audit trail

## Architecture

```
Razorpay Test Mode
      │ webhooks (payment.failed, payment_link.paid)
      ▼
FastAPI backend ──► PostgreSQL (transactions, decisions, audit trail)
      │
      ▼
XGBoost Recovery Predictor → Decision Engine (expected value)
      │
      ▼
Policy Engine (auto-execute vs. human approval)
      │
      ▼
Razorpay Payment Link API (real action)
      │
      ▼
Dashboard (live, auto-refreshing)
```

## Tech Stack

- **Backend:** Python, FastAPI, SQLAlchemy, PostgreSQL
- **ML:** XGBoost, scikit-learn, pandas
- **AI reasoning:** Claude (Anthropic API) — explains decisions, never makes them
- **Payments:** Razorpay Test Mode (Payment Links + Webhooks)
- **Frontend:** Vanilla HTML/CSS/JS (no framework — kept deliberately simple)

## Model Evaluation (held-out test set, honestly reported)

| Metric | Score |
|---|---|
| ROC-AUC | 0.634 |
| Precision | 0.573 |
| Recall | 0.605 |
| F1 | 0.588 |

This is a synthetic dataset (2,500 transactions) with deliberately realistic — not perfectly separable — patterns: returning customers recover at a meaningfully higher rate than new customers, and `payment_link` outperforms `do_nothing`, but neither is a guarantee. We report this as-is rather than tuning for a better-looking number.

## Setup Instructions

### Prerequisites
- Python 3.10+
- PostgreSQL
- A Razorpay account (Test Mode)
- An Anthropic API key (optional — falls back to a templated explanation if not provided)

### Steps

```bash
# 1. Clone and enter the repo
git clone <your-repo-url>
cd revive-ai-revenue-recovery-agent

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file with:
#   DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/your_db
#   RAZORPAY_KEY_ID=your_test_key_id
#   RAZORPAY_KEY_SECRET=your_test_key_secret
#   RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
#   ANTHROPIC_API_KEY=your_anthropic_key   (optional)

# 5. Create the database tables
python models.py

# 6. Generate synthetic training data and train the model
python generate_dataset.py --rows 2500 --seed 42 --out transactions.csv
python train_model.py

# 7. Run the server
uvicorn main:app --reload

# 8. In a separate terminal, expose it publicly for Razorpay webhooks
ngrok http 8000

# 9. Register the ngrok URL + /webhooks/razorpay in Razorpay Dashboard →
#    Settings → Webhooks, subscribed to: payment.failed, payment_link.paid

# 10. Open the dashboard
# http://127.0.0.1:8000/dashboard
```

## What Broke at 2 AM (and how we got out)

Being upfront about this because it's a more honest picture of the build than a polished demo alone:

**Schema drift, repeatedly.** Every time we added a new column to the database model (linking payments to payment links, then adding decision tracking, then approval status), the actual PostgreSQL table didn't update automatically — SQLAlchemy only creates tables that don't exist yet, it doesn't migrate existing ones. We hit `UndefinedColumn` errors three separate times before building a proper fix: `models.py` now inspects the live table on startup, diffs it against what the code expects, and prompts before dropping/recreating — so this class of bug can't silently reappear.

**Duplicate transaction rows from overlapping webhooks.** Subscribing to both `payment.captured` and `payment_link.paid` meant Razorpay sent two separate events for the same successful payment, creating two database rows for one real transaction. Fixed by unsubscribing from `payment.captured` entirely — since Payment Link is our only real intervention, `payment_link.paid` alone is sufficient and unambiguous.

**Stale local files silently diverging from the canonical version.** Column names got renamed mid-build (`time_since_failure_hrs` → `hours_since_failure`, etc.) but a downloaded copy of an older file lingered in a `Downloads` folder and got re-used, causing `KeyError`s that looked like new bugs but were actually just two files disagreeing on a schema that had already been fixed elsewhere. Resolved by pasting canonical file contents directly rather than relying on repeated downloads once this started happening.

**ngrok tunnel dying silently between work sessions.** The tunnel doesn't survive a terminal closing, and there's no obvious error on the Razorpay side — webhooks just stop arriving, which initially looked like a backend bug. Now treated as a standard startup checklist item (three terminals always running: backend, tunnel, scratch) rather than something to debug after the fact.

## Known Limitations / Future Work

- Single intervention type (Payment Link only) — Reminder, Retry, and Discount were scoped out to keep one workflow fully real rather than several partially simulated
- No production-grade auth on the approval endpoints (fine for Test Mode; would need it before handling real merchant data)
- Customer history features are placeholder values on live webhook data (no real historical lookup yet) — the model itself is trained on realistic synthetic history
- Human approval UI is functional but minimal — no audit log of *who* approved, just that it happened

## Team / Submission

Built for Razorpay's Buildathon — Track 03: AI Revenue Recovery.
