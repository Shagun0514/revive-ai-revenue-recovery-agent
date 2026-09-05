"""
Revive — AI Revenue Recovery Agent
Database models.

Run this file once to create the tables:
    python models.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True)          # Razorpay payment_id or order_id
    event_type = Column(String)                     # payment.failed / payment.captured / payment_link.paid
    amount = Column(Float)
    currency = Column(String, default="INR")
    status = Column(String)                          # failed / captured / link_created
    payment_link_id = Column(String, nullable=True)
    payment_link_url = Column(String, nullable=True)
    payment_id = Column(String, nullable=True)        # the actual payment ID, once paid via a link
    recovered = Column(Boolean, default=False)
    recommended_action = Column(String, nullable=True)  # do_nothing / payment_link
    expected_profit = Column(Float, nullable=True)
    approval_status = Column(String, default="not_applicable")  # auto_approved / pending_approval / approved / rejected / not_applicable
    llm_explanation = Column(String, nullable=True)
    raw_payload = Column(String, nullable=True)       # store full JSON as text, handy for debugging
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


if __name__ == "__main__":
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. 'transactions' table is ready.")
