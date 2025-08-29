import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from utils.db import SessionLocal, RLWeights, ExplainLog, get_or_create_weights

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

WEIGHTS_CSV = os.path.join(MODEL_DIR, "rl_weights.csv")
REWARD_PLOT = os.path.join(LOG_DIR, "reward_curve.png")

# configurable interval (default: 5 updates)
REWARD_SAVE_INTERVAL = int(os.getenv("RL_REWARD_SAVE_INTERVAL", "5"))

class RLCandidateAgent:
    """
    DB-backed RL agent.
    """

    def __init__(self, lr: float = 0.1):
        self.lr = float(lr)
        self.session = SessionLocal()
        w = get_or_create_weights(self.session)
        self.alpha, self.beta, self.gamma = float(w.alpha), float(w.beta), float(w.gamma)
        
        # ✅ reward history and updates tracker
        self.rewards = []
        self.reward_history = []
        self.total_updates = 0
        
        self.weights_history = [(self.alpha, self.beta, self.gamma)]

    def _persist_weights(self):
        """Persist weights to DB and CSV."""
        w = self.session.query(RLWeights).first()
        if not w:
            w = RLWeights(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
            self.session.add(w)
        else:
            w.alpha, w.beta, w.gamma = float(self.alpha), float(self.beta), float(self.gamma)
        self.session.commit()

        header_needed = not os.path.exists(WEIGHTS_CSV)
        with open(WEIGHTS_CSV, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("ts,alpha,beta,gamma\n")
            f.write(f"{datetime.utcnow().isoformat()},{self.alpha:.6f},{self.beta:.6f},{self.gamma:.6f}\n")

    def score(self, original_match: float, sentiment: float, values: float) -> float:
        """Compute combined score (alpha*match + beta*sentiment + gamma*values)."""
        return float(
            self.alpha * float(original_match) +
            self.beta * float(sentiment) +
            self.gamma * float(values)
        )

    def update(self, original_match: float, sentiment: float, values: float,
               action: str, feedback_sentiment: str):
        """
        Update weights based on reward signal from recruiter feedback.
        Returns reward (float).
        """
        s = (feedback_sentiment or "").strip().lower()
        ideal_action = "shortlist" if s == "positive" else ("reject" if s == "negative" else "neutral")
        
        if ideal_action == "neutral":
            reward = 0.0
        else:
            reward = 1.0 if action == ideal_action else -1.0

        # ✅ track rewards and updates
        self.rewards.append(float(reward))
        self.reward_history.append({
            "update": self.total_updates,
            "reward": float(reward),
            "cumulative": float(np.cumsum(self.rewards)[-1])
        })
        self.total_updates += 1

        # Gradient-like update
        self.alpha += self.lr * float(reward) * float(original_match)
        self.beta  += self.lr * float(reward) * float(sentiment)
        self.gamma += self.lr * float(reward) * float(values)

        # Prevent negatives & normalize
        self.alpha, self.beta, self.gamma = [max(0.0, w) for w in (self.alpha, self.beta, self.gamma)]
        total = max(1e-9, self.alpha + self.beta + self.gamma)
        self.alpha, self.beta, self.gamma = [w / total for w in (self.alpha, self.beta, self.gamma)]

        self.weights_history.append((self.alpha, self.beta, self.gamma))
        self._persist_weights()

        # 🔥 Auto-save reward curve every N updates
        if REWARD_SAVE_INTERVAL > 0 and len(self.rewards) % REWARD_SAVE_INTERVAL == 0:
            self.save_reward_curve()

        return float(reward)

    def log_explain(self, candidate_id: str, action: str, reward: float,
                    sentiment_label: str, sentiment_score: float,
                    values_score: float, final_score: float, reason_summary: str):
        """Persist explain record to DB."""
        rec = ExplainLog(
            candidate_id=str(candidate_id),
            action=str(action),
            reward=float(reward),
            sentiment_label=str(sentiment_label),
            sentiment_score=float(sentiment_score),
            values_score=float(values_score),
            final_score=float(final_score),
            weights=f"alpha={self.alpha:.4f},beta={self.beta:.4f},gamma={self.gamma:.4f}",
            reason_summary=str(reason_summary)
        )
        self.session.add(rec)
        self.session.commit()

    def get_weights(self):
        return {
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "gamma": round(self.gamma, 6)
        }

    def save_reward_curve(self, path: str = REWARD_PLOT):
        """Plot cumulative reward curve to PNG."""
        if not self.rewards:
            return
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(np.cumsum(self.rewards), marker="o")
            plt.title("RL Cumulative Reward")
            plt.xlabel("Iteration")
            plt.ylabel("Cumulative Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception:
            pass


