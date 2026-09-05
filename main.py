"""
Revive — AI Revenue Recovery Agent
Step 8: Minimal starter to confirm the environment works end to end.

Run with:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""

import os
import hmac
import hashlib
import json

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import razorpay

from models import SessionLocal, Transaction
from decision_engine import evaluate_transaction
from llm_reasoning import generate_explanation

# Load variables from .env into the environment
load_dotenv()

app = FastAPI(title="Revive — AI Revenue Recovery Agent")

DATABASE_URL = os.getenv("DATABASE_URL")
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# --- Merchant policy (hardcoded for MVP; move to DB/config later if time allows) ---
POLICY = {
    "payment_link_enabled": True,
    "auto_execute_under": 10000,   # amounts below this: agent acts automatically
    "approval_required_above": 10000,  # amounts at/above this: flagged, NOT auto-executed
}

app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Revive backend is alive.",
    }


@app.get("/health")
def health_check():
    """
    Checks that:
    1. .env variables are loading correctly
    2. The database connection actually works
    """
    result = {
        "env_loaded": DATABASE_URL is not None,
        "razorpay_key_loaded": RAZORPAY_KEY_ID is not None,
        "database_connected": False,
    }

    if DATABASE_URL:
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            result["database_connected"] = True
        except Exception as e:
            result["database_error"] = str(e)

    return result


def verify_razorpay_signature(payload_body: bytes, received_signature: str) -> bool:
    """
    Razorpay signs every webhook with your Webhook Secret so you can confirm
    the request genuinely came from Razorpay and wasn't sent by someone else.
    We recompute the signature ourselves and compare.
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        # No secret configured yet — allow through during initial local testing only.
        return True

    expected_signature = hmac.new(
        key=RAZORPAY_WEBHOOK_SECRET.encode(),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_signature, received_signature)


@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")

    if not verify_razorpay_signature(raw_body, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    payload = json.loads(raw_body)
    event_type = payload.get("event", "unknown")

    print("=" * 60)
    print(f"WEBHOOK RECEIVED: {event_type}")
    print(json.dumps(payload, indent=2))
    print("=" * 60)

    db = SessionLocal()
    try:
        if event_type == "payment_link.paid":
            # This event bundles BOTH the payment link and the payment together —
            # use it to update the existing payment-link row instead of creating a new one.
            link_entity = payload.get("payload", {}).get("payment_link", {}).get("entity", {})
            payment_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})

            link_id = link_entity.get("id")
            payment_id = payment_entity.get("id")

            if link_id:
                existing = db.get(Transaction, link_id)
                if existing:
                    existing.event_type = event_type
                    existing.status = "paid"
                    existing.recovered = True
                    existing.payment_id = payment_id
                    existing.raw_payload = json.dumps(payload)
                    db.commit()

        else:
            # payment.failed / payment.captured (not necessarily via a payment link)
            entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
            payment_id = entity.get("id")

            if payment_id:
                # Case 1: this payment_id IS the row's primary key (a standalone failed/captured event)
                existing = db.get(Transaction, payment_id)

                # Case 2: this payment_id was already linked to a payment-link row
                already_linked = None
                if not existing:
                    already_linked = (
                        db.query(Transaction)
                        .filter(Transaction.payment_id == payment_id)
                        .first()
                    )

                if existing:
                    existing.event_type = event_type
                    existing.status = entity.get("status", event_type)
                    existing.recovered = (event_type == "payment.captured")
                    existing.raw_payload = json.dumps(payload)
                    db.commit()
                elif already_linked:
                    pass
                else:
                    amount_rupees = (entity.get("amount", 0) or 0) / 100
                    new_txn = Transaction(
                        id=payment_id,
                        event_type=event_type,
                        amount=amount_rupees,
                        currency=entity.get("currency", "INR"),
                        status=entity.get("status", event_type),
                        recovered=(event_type == "payment.captured"),
                        raw_payload=json.dumps(payload),
                    )
                    db.add(new_txn)
                    db.commit()

                    # --- AGENT DECISION: only for failed payments ---
                    if event_type == "payment.failed" and amount_rupees > 0:
                        txn_features = {
                            "amount": amount_rupees,
                            "attempt_count": 1,             # we don't have real history yet — placeholder
                            "hours_since_failure": 0.1,
                            "previous_transactions": 0,
                            "previous_success_rate": 0.5,
                            "is_repeat_customer": 0,
                            "payment_method": entity.get("method", "upi"),
                            "customer_type": "new",
                            "failure_reason": entity.get("error_reason", "temporary_failure") or "temporary_failure",
                            "merchant_category": "ecommerce",
                        }

                        decision = evaluate_transaction(txn_features)
                        recommended = decision["recommended_action"]

                        print("-" * 60)
                        print(f"AGENT DECISION for {payment_id}")
                        print(json.dumps(decision, indent=2))

                        explanation = generate_explanation(txn_features, decision)
                        print(f"EXPLANATION: {explanation}")

                        # --- POLICY CHECK ---
                        policy_approved = (
                            POLICY["payment_link_enabled"]
                            and recommended == "payment_link"
                            and amount_rupees < POLICY["auto_execute_under"]
                        )

                        new_txn.recommended_action = recommended
                        new_txn.expected_profit = decision["options"][recommended]["expected_profit"]
                        new_txn.llm_explanation = explanation

                        if recommended != "payment_link":
                            new_txn.approval_status = "not_applicable"
                            print("POLICY: no action needed (do_nothing was the better option)")

                        elif policy_approved:
                            new_txn.approval_status = "auto_approved"
                            print(f"POLICY: APPROVED — auto-creating payment link for ₹{amount_rupees}")
                            try:
                                link = razorpay_client.payment_link.create({
                                    "amount": int(amount_rupees * 100),
                                    "currency": "INR",
                                    "description": "Complete your payment — Revive Autopilot",
                                    "customer": {
                                        "name": entity.get("customer_id", "Customer") or "Customer",
                                        "email": entity.get("email", "customer@example.com") or "customer@example.com",
                                        "contact": entity.get("contact", "9876543210") or "9876543210",
                                    },
                                    "notify": {"sms": False, "email": False},
                                    "reminder_enable": False,
                                })
                                new_txn.payment_link_id = link["id"]
                                new_txn.payment_link_url = link["short_url"]
                                new_txn.status = "recovery_link_sent"
                                print(f"Payment link created: {link['short_url']}")
                            except Exception as e:
                                print(f"Payment link creation FAILED: {e}")

                        else:
                            # Recommended payment_link, but amount is above the auto-execute threshold —
                            # hold for human approval instead of silently skipping.
                            new_txn.approval_status = "pending_approval"
                            print(f"POLICY: HELD — ₹{amount_rupees} exceeds auto-execute threshold, needs human approval")

                        db.commit()
                        print("-" * 60)
    finally:
        db.close()

    return {"status": "received", "event": event_type}


class CreatePaymentLinkRequest(BaseModel):
    amount: float          # in rupees, e.g. 500.00
    description: str = "Complete your payment"
    customer_name: str = "Customer"
    customer_email: str = "customer@example.com"
    customer_contact: str = "9999999999"


@app.post("/payment-links")
def create_payment_link(req: CreatePaymentLinkRequest):
    """
    Creates a real Razorpay Test Mode payment link and returns its URL.
    This is the ONE real recovery action in the MVP scope.
    """
    try:
        link = razorpay_client.payment_link.create({
            "amount": int(req.amount * 100),  # Razorpay expects paise
            "currency": "INR",
            "description": req.description,
            "customer": {
                "name": req.customer_name,
                "email": req.customer_email,
                "contact": req.customer_contact,
            },
            "notify": {"sms": False, "email": False},
            "reminder_enable": False,
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save it to the database so we can track it as a recovery attempt
    db = SessionLocal()
    try:
        txn = Transaction(
            id=link["id"],
            event_type="payment_link.created",
            amount=req.amount,
            status="link_created",
            payment_link_id=link["id"],
            payment_link_url=link["short_url"],
            raw_payload=json.dumps(link),
        )
        db.merge(txn)  # merge = insert or update if id already exists
        db.commit()
    finally:
        db.close()

    return {
        "payment_link_id": link["id"],
        "payment_link_url": link["short_url"],
        "status": link["status"],
    }


@app.post("/approve/{transaction_id}")
def approve_transaction(transaction_id: str):
    """Human approves a payment link that was held above the auto-execute threshold."""
    db = SessionLocal()
    try:
        txn = db.get(Transaction, transaction_id)
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")
        if txn.approval_status != "pending_approval":
            raise HTTPException(status_code=400, detail=f"Transaction is '{txn.approval_status}', not pending approval")

        try:
            link = razorpay_client.payment_link.create({
                "amount": int(txn.amount * 100),
                "currency": "INR",
                "description": "Complete your payment — Revive Autopilot (approved)",
                "customer": {
                    "name": "Customer",
                    "email": "customer@example.com",
                    "contact": "9876543210",
                },
                "notify": {"sms": False, "email": False},
                "reminder_enable": False,
            })
            txn.payment_link_id = link["id"]
            txn.payment_link_url = link["short_url"]
            txn.status = "recovery_link_sent"
            txn.approval_status = "approved"
            db.commit()
            return {"status": "approved", "payment_link_url": link["short_url"]}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@app.post("/reject/{transaction_id}")
def reject_transaction(transaction_id: str):
    """Human rejects — agent will not create a payment link for this transaction."""
    db = SessionLocal()
    try:
        txn = db.get(Transaction, transaction_id)
        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")
        txn.approval_status = "rejected"
        db.commit()
        return {"status": "rejected"}
    finally:
        db.close()


@app.get("/dashboard-data")
def dashboard_data():
    db = SessionLocal()
    try:
        all_txns = db.query(Transaction).order_by(Transaction.created_at.desc()).all()

        failed_txns = [t for t in all_txns if t.event_type in ("payment.failed",) or t.status == "failed"]
        revenue_at_risk = sum(t.amount or 0 for t in failed_txns)

        recovered_txns = [t for t in all_txns if t.recovered]
        revenue_recovered = sum(t.amount or 0 for t in recovered_txns)

        expected_recoverable = sum(t.expected_profit or 0 for t in all_txns if t.expected_profit)

        recovery_rate = (len(recovered_txns) / len(failed_txns) * 100) if failed_txns else 0

        pending = [t for t in all_txns if t.approval_status == "pending_approval"]
        pending_approvals = [{
            "id": t.id,
            "amount": t.amount,
            "expected_profit": t.expected_profit,
            "llm_explanation": t.llm_explanation,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        } for t in pending]

        recent = []
        for t in all_txns[:20]:
            recent.append({
                "id": t.id,
                "event_type": t.event_type,
                "amount": t.amount,
                "status": t.status,
                "recommended_action": t.recommended_action,
                "expected_profit": t.expected_profit,
                "approval_status": t.approval_status,
                "llm_explanation": t.llm_explanation,
                "recovered": t.recovered,
                "payment_link_url": t.payment_link_url,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            })

        return {
            "revenue_at_risk": round(revenue_at_risk, 2),
            "expected_recoverable": round(expected_recoverable, 2),
            "revenue_recovered": round(revenue_recovered, 2),
            "recovery_rate": round(recovery_rate, 1),
            "total_failed": len(failed_txns),
            "total_recovered": len(recovered_txns),
            "pending_approvals": pending_approvals,
            "recent_transactions": recent,
        }
    finally:
        db.close()
