"""
Revive — AI Revenue Recovery Agent
LLM Reasoning Layer.

IMPORTANT: this module only EXPLAINS decisions already made by the
decision engine (decision_engine.py). It never makes the financial
decision itself — that stays deterministic and auditable.
"""

import os
import anthropic

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_explanation(txn: dict, decision: dict) -> str:
    """
    txn: the transaction features dict (same one passed to evaluate_transaction)
    decision: the dict returned by decision_engine.evaluate_transaction()

    Returns a short, merchant-facing explanation string.
    If no API key is configured or the call fails, returns a safe fallback
    so the webhook never breaks because of this optional layer.
    """
    client = _get_client()
    if client is None:
        return _fallback_explanation(decision)

    recommended = decision["recommended_action"]
    options = decision["options"]

    prompt = f"""A payment recovery AI agent just evaluated a failed transaction and made a decision.

Transaction:
- Amount: ₹{txn['amount']:.0f}
- Payment method: {txn['payment_method']}
- Customer type: {txn['customer_type']}
- Failure reason: {txn['failure_reason']}

Decision options evaluated:
- Do nothing: {options['do_nothing']['recovery_probability']*100:.0f}% recovery probability, ₹{options['do_nothing']['expected_profit']:.0f} expected profit
- Send payment link: {options['payment_link']['recovery_probability']*100:.0f}% recovery probability, ₹{options['payment_link']['expected_profit']:.0f} expected profit

Recommended action: {recommended}

Write a 2-sentence, plain-language explanation for a merchant (non-technical) of why this action was recommended. Be concrete about the numbers. No preamble, just the explanation."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"LLM explanation failed (non-fatal): {e}")
        return _fallback_explanation(decision)


def _fallback_explanation(decision: dict) -> str:
    """Used if no API key is set, or the API call fails — keeps the agent working regardless."""
    recommended = decision["recommended_action"]
    profit = decision["options"][recommended]["expected_profit"]
    prob = decision["options"][recommended]["recovery_probability"]
    action_label = "sending a payment link" if recommended == "payment_link" else "not intervening"
    return (
        f"The agent recommends {action_label}, with an estimated "
        f"{prob*100:.0f}% recovery probability and ₹{profit:.0f} expected profit."
    )
