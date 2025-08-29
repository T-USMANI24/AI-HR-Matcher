# agent.py (Final – Unique Candidate IDs + RL Status Endpoint)
import os
import json
import uuid
import hashlib
from flask import Flask, request, jsonify
from datetime import datetime
from rapidfuzz import fuzz
import numpy as np
from utils.preprocess import (
    prepare_document, clean_text_for_model, extract_skills,
    extract_degrees, extract_experience_years, clean_feedback
)
from utils.embedding import compute_similarity, embed_feedback
from utils.sentiment import get_feedback_sentiment
from utils.rl_agent import RLCandidateAgent
from utils.db import init_db, SessionLocal, ExplainLog

# -----------------------------
# Directories & Logs
# -----------------------------
LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MATCH_LOG = os.path.join(LOG_DIR, "agent_match_log.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "agent_feedback.jsonl")

# Init DB + RL agent
app = Flask(__name__)
init_db()
rl_agent = RLCandidateAgent(lr=float(os.getenv("RL_LR", "0.1")))

# -----------------------------
# Helper Functions
# -----------------------------
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def fuzzy_skill_match(cv_skills, jd_skills, threshold=75):
    matched = []
    for jd in jd_skills:
        for cv_s in cv_skills:
            if fuzz.ratio(cv_s.lower(), jd.lower()) >= threshold:
                matched.append(jd)
                break
    return sorted(set(matched))

def mdvp_reflection():
    return {
        "humility": "Listened to recruiter feedback; tuned weights transparently.",
        "gratitude": "Thankful for human-in-the-loop guidance that shapes better hiring.",
        "integrity": "Logged every recommendation with explainability and versioned weights."
    }

def compute_values_score(values_dict, weights=None):
    if weights is None:
        weights = {"integrity":0.25, "honesty":0.25, "discipline":0.25, "hard_work":0.25}
    max_score = sum(weights.values()) * 5
    total = sum(values_dict.get(k,0) * weights.get(k,0) for k in weights)
    return total / max_score if max_score > 0 else 0.0

def generate_candidate_id(cv_dict, idx):
    """
    Generate stable unique ID for each CV:
    - Use provided ID (file name from frontend) if exists
    - Else hash of text content + uuid
    """
    if cv_dict.get("id"):
        return str(cv_dict["id"])
    text = (cv_dict.get("text") or "").encode("utf-8")
    h = hashlib.md5(text).hexdigest()[:6]
    return f"cv{idx+1}_{h}_{uuid.uuid4().hex[:4]}"

# -----------------------------
# POST /agent/match
# -----------------------------
@app.route("/agent/match", methods=["POST"])
def match():
    payload = request.get_json(force=True)
    jd_text = payload.get("jd_text") or payload.get("jd", "")
    cvs = payload.get("cvs", [])
    method = payload.get("method", None)

    if not jd_text or not cvs:
        return jsonify({"error": "jd_text and cvs required"}), 400

    cv_texts = []
    for c in cvs:
        txt = c.get("text") or ""
        if not txt and c.get("path"):
            txt = prepare_document(c.get("path"))
        cv_texts.append(txt)

    sims = compute_similarity(cv_texts, jd_text, method=method)
    jd_skills = extract_skills(jd_text)
    jd_degrees = extract_degrees(jd_text)

    results = []
    for i, c in enumerate(cvs):
        cid = generate_candidate_id(c, i)
        text = cv_texts[i] or ""
        parsed_skills = extract_skills(text)
        experience = extract_experience_years(text)
        matched_skills = fuzzy_skill_match(parsed_skills, jd_skills)
        skill_pct = round(len(matched_skills) / max(1, len(jd_skills)), 2) if jd_skills else 0.0
        sim_score = float(sims[i])
        values_score = compute_values_score(c.get("values", {}))

        combined = round(
            skill_pct * 0.6 + sim_score * 0.3 + min(1.0, experience/5.0) * 0.1,
            3
        )

        reason = {
            "matched_skills": matched_skills,
            "missing_skills": sorted(set(jd_skills) - set(matched_skills)),
            "parsed_skills": parsed_skills,
            "similarity_score": round(sim_score, 4),
            "skill_pct": skill_pct,
            "experience_years": experience,
            "degree": extract_degrees(text)
        }

        result = {
            "candidate_id": cid,
            "combined_score": combined,
            "similarity_score": round(sim_score, 4),
            "skill_score": skill_pct,
            "experience_years": experience,
            "degree_match": bool(set(extract_degrees(text)) & set(jd_degrees)) or ("ANY" in jd_degrees),
            "values_score": round(values_score, 3),
            "sentiment_score": None,
            "reason": reason
        }
        results.append(result)

        append_jsonl(MATCH_LOG, {
            "ts": datetime.utcnow().isoformat(),
            "candidate_id": cid,
            "jd_snippet": jd_text[:300],
            "cv_snippet": (text or "")[:300],
            "result": result
        })

    return jsonify({"results": results, "mdvp": mdvp_reflection()})

# -----------------------------
# POST /agent/feedback
# -----------------------------
@app.route("/agent/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(force=True)
    candidate_id = payload.get("candidate_id")

    if not candidate_id:
        candidate_id = f"fb_{uuid.uuid4().hex[:6]}"

    raw_fb = payload.get("feedback_text", "")
    cleaned = clean_feedback(raw_fb)
    fb_emb = embed_feedback(raw_fb)
    if fb_emb is not None and not isinstance(fb_emb, list):
        fb_emb = np.array(fb_emb).tolist()

    sentiment_label, sentiment_score = get_feedback_sentiment(raw_fb)
    values_score = compute_values_score(payload.get("values", {}))
    original_match = float(payload.get("original_match", 0))

    final_score = rl_agent.score(original_match, sentiment_score, values_score)
    if final_score >= 0.45:
        action = "shortlist"
    elif final_score <= 0.35:
        action = "reject"
    else:
        action = "neutral"

    reward = rl_agent.update(original_match, sentiment_score, values_score, action, sentiment_label)

    explain_summary = (
        f"action='{action}', feedback_sentiment='{sentiment_label}', "
        f"reward={reward}, weights α={rl_agent.alpha:.3f}, β={rl_agent.beta:.3f}, γ={rl_agent.gamma:.3f}"
    )

    rl_agent.log_explain(
        candidate_id=candidate_id,
        action=action,
        reward=reward,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        values_score=values_score,
        final_score=final_score,
        reason_summary=explain_summary
    )

    feedback_record = {
        "ts": datetime.utcnow().isoformat(),
        "candidate_id": candidate_id,
        "recruiter_id": payload.get("recruiter_id", "anonymous"),
        "feedback_text": raw_fb,
        "cleaned_feedback": cleaned,
        "feedback_embedding": fb_emb,
        "values": payload.get("values", {}),
        "sentiment_label": sentiment_label,
        "sentiment_score": round(sentiment_score, 3),
        "values_score": round(values_score, 3),
        "original_match": round(original_match, 3),
        "final_score": round(final_score, 3),
        "action": action,
        "reward": reward,
        "weights": rl_agent.get_weights(),
        "mdvp": mdvp_reflection(),
        "explain_summary": explain_summary
    }
    append_jsonl(FEEDBACK_LOG, feedback_record)

    return jsonify({"ok": True, "saved": feedback_record})

# -----------------------------
# GET /agent/explain/<candidate_id>
# -----------------------------
@app.route("/agent/explain/<candidate_id>", methods=["GET"])
def explain(candidate_id):
    session = SessionLocal()
    try:
        rows = (
            session.query(ExplainLog)
            .filter(ExplainLog.candidate_id == candidate_id)
            .order_by(ExplainLog.ts.desc())
            .limit(50)
            .all()
        )

        timeline = []
        for r in rows:
            ts_str = r.ts.strftime("%d %b %Y – %H:%M")
            symbol = "✅" if r.action == "shortlist" else ("❌" if r.action == "reject" else "➖")

            reason = (
                f"Candidate final score {r.final_score:.2f} | "
                f"values score {r.values_score:.2f} | "
                f"recruiter feedback {r.sentiment_label} ({r.sentiment_score:.2f})"
            )

            if float(r.reward) == 1.0:
                reason += " → agent’s decision matched recruiter ✅"
            elif float(r.reward) == -1.0:
                reason += " → agent’s decision conflicted with recruiter ❌"
            else:
                reason += " → no learning reward applied"

            entry = f"[{ts_str}]\nDecision: {r.action.capitalize()} {symbol}\nReason: {reason}"
            timeline.append(entry)

        return jsonify({
            "candidate_id": candidate_id,
            "timeline": timeline
        })
    finally:
        session.close()

# -----------------------------
# GET /agent/mdvp
# -----------------------------
@app.route("/agent/mdvp", methods=["GET"])
def mdvp():
    return jsonify(mdvp_reflection())

# -----------------------------
# NEW: GET /agent/status
# -----------------------------
@app.route("/agent/status", methods=["GET"])
def status():
    """Return RL agent training status (weights + reward history + accuracy)."""

    reward_history = rl_agent.reward_history[-50:]  # last 50 rewards
    total = sum(1 for r in reward_history if r != 0)
    correct = sum(1 for r in reward_history if r == 1)
    accuracy = (correct / total * 100) if total > 0 else 0.0

    return jsonify({
        "weights": rl_agent.get_weights(),
        "total_updates": rl_agent.total_updates,
        "reward_history": reward_history,
        "cumulative_reward": float(np.cumsum(rl_agent.rewards)[-1]) if rl_agent.rewards else 0.0,
        "accuracy": accuracy  # 🔥 now backend provides accuracy
    })


print("🔍 Registered routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)



