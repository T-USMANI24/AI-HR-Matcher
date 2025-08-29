# db.py
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text
)
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///models/agent.db")

# For sqlite, need check_same_thread=False
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class RLWeights(Base):
    __tablename__ = "rl_weights"
    id = Column(Integer, primary_key=True, index=True)
    alpha = Column(Float, nullable=False, default=0.6)
    beta  = Column(Float, nullable=False, default=0.2)
    gamma = Column(Float, nullable=False, default=0.2)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExplainLog(Base):
    __tablename__ = "explain_log"
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime, default=datetime.utcnow, index=True)
    candidate_id = Column(String(128), index=True)
    action = Column(String(32))
    reward = Column(Float)
    sentiment_label = Column(String(32))
    sentiment_score = Column(Float)
    values_score = Column(Float)
    final_score = Column(Float)
    weights = Column(String(256))  # e.g. "alpha=..,beta=..,gamma=.."
    reason_summary = Column(Text)

def init_db():
    """Create DB tables if not already present."""
    Base.metadata.create_all(bind=engine)

def get_or_create_weights(session):
    """Return first RLWeights row; create default if absent."""
    obj = session.query(RLWeights).first()
    if not obj:
        obj = RLWeights(alpha=0.6, beta=0.2, gamma=0.2)
        session.add(obj)
        session.commit()
        session.refresh(obj)
    return obj
