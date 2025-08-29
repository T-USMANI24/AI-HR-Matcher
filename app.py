import streamlit as st
import requests
import json
import pandas as pd
import fitz  # PyMuPDF for PDF reading
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Candidate Matching Agent", layout="wide")

st.title("🤖 Candidate Matching Agent")
st.write("Upload JDs and CVs, give feedback, and see how the RL agent adapts.")

# ---------------------------
# Sidebar – Navigation
# ---------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Match Candidates", "Submit Feedback", "Explainability Logs", "Training Monitor", "MDVP Reflections"]
)

# ---------------------------
# Helper – Extract Text
# ---------------------------
def extract_text_from_file(uploaded_file):
    """Reads text from TXT or PDF file."""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    else:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text

# ---------------------------
# Match Candidates
# ---------------------------
if page == "Match Candidates":
    st.header("📄 Match Candidates to a Job Description")

    jd_file = st.file_uploader("Upload Job Description (txt or pdf)", type=["txt", "pdf"])
    jd_text_area = st.text_area("Or paste JD text here:", height=150)

    cv_files = st.file_uploader(
        "Upload CVs (txt or pdf, multiple allowed)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )
    cv_text_area = st.text_area(
        "Or paste CVs (separate with '---')",
        placeholder="Candidate 1 CV...\n---\nCandidate 2 CV..."
    )

    method = st.selectbox("Similarity Method", ["tfidf", "sbert"], index=0)

    if st.button("Run Match"):
        jd_text = ""
        if jd_file is not None:
            jd_text = extract_text_from_file(jd_file)
        elif jd_text_area.strip():
            jd_text = jd_text_area.strip()

        cvs = []
        if cv_files:
            for idx, cvf in enumerate(cv_files):
                text = extract_text_from_file(cvf)
                file_name = os.path.splitext(cvf.name)[0]  # filename → candidate_id
                cvs.append({
                    "id": file_name,
                    "text": text,
                    "values": {"integrity": 4, "honesty": 4, "discipline": 3, "hard_work": 4}
                })
        elif cv_text_area.strip():
            for idx, cv in enumerate(cv_text_area.split("---")):
                cvs.append({
                    "id": f"cv{idx+1}",
                    "text": cv.strip(),
                    "values": {"integrity": 4, "honesty": 4, "discipline": 3, "hard_work": 4}
                })

        if not jd_text or not cvs:
            st.warning("Please provide a JD and at least one CV.")
        else:
            payload = {"jd_text": jd_text, "cvs": cvs, "method": method}
            resp = requests.post(f"{API_URL}/agent/match", json=payload)
            if resp.status_code == 200:
                results = resp.json()
                st.session_state["last_results"] = results["results"]

                st.subheader("Results Table")
                df = pd.DataFrame(results["results"])
                st.dataframe(df)

                st.subheader("Scores Chart")
                st.bar_chart(df.set_index("candidate_id")["combined_score"])

                st.download_button(
                    "⬇️ Download Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name="match_results.json",
                    mime="application/json"
                )
                st.download_button(
                    "⬇️ Download Results (CSV)",
                    data=df.to_csv(index=False),
                    file_name="match_results.csv",
                    mime="text/csv"
                )

                st.success("✅ Matching complete!")
            else:
                st.error(f"Error: {resp.text}")

# ---------------------------
# Submit Feedback
# ---------------------------
elif page == "Submit Feedback":
    st.header("📝 Submit Recruiter Feedback")

    candidate_id = None
    original_match = 0.0

    if "last_results" in st.session_state:
        options = {c["candidate_id"]: c["combined_score"] for c in st.session_state["last_results"]}
        candidate_id = st.selectbox("Select Candidate", list(options.keys()))
        if candidate_id:
            original_match = options[candidate_id]
    else:
        candidate_id = st.text_input("Candidate ID (e.g., cv1):")
        original_match = st.number_input("Original Match Score (0–1)", value=0.7, step=0.01)

    feedback_text = st.text_area("Feedback Text:", placeholder="Strong technical skills...")

    integrity = st.slider("Integrity", 0, 5, 4, help="Consistency & truthfulness")
    honesty = st.slider("Honesty", 0, 5, 4, help="Transparency & genuineness")
    discipline = st.slider("Discipline", 0, 5, 3, help="Following deadlines & processes")
    hard_work = st.slider("Hard Work", 0, 5, 5, help="Consistency of effort")

    if candidate_id:
        st.info(f"Original Match Score for {candidate_id}: **{original_match:.2f}**")

    if st.button("Submit Feedback"):
        if candidate_id and feedback_text:
            payload = {
                "candidate_id": candidate_id,
                "feedback_text": feedback_text,
                "values": {
                    "integrity": integrity,
                    "honesty": honesty,
                    "discipline": discipline,
                    "hard_work": hard_work
                },
                "original_match": original_match
            }
            resp = requests.post(f"{API_URL}/agent/feedback", json=payload)
            if resp.status_code == 200:
                saved = resp.json()["saved"]
                st.subheader("Saved Feedback")
                st.json(saved)
                st.success("✅ Feedback submitted!")

                # 🔥 Fetch latest RL agent status
                status_resp = requests.get(f"{API_URL}/agent/status")
                if status_resp.status_code == 200:
                    status = status_resp.json()

                    st.subheader("📈 RL Agent Reward Curve")
                    st.line_chart(status["reward_history"])

                    st.subheader("Weights (α, β, γ)")
                    st.json(status["weights"])

                    st.metric("Total Updates", status["total_updates"])
                    st.metric("Cumulative Reward", round(status["cumulative_reward"], 2))

                    if status["reward_history"]:
                        # Only count non-neutral rewards
                        total = sum(1 for r in status["reward_history"] if r != 0)
                        correct = sum(1 for r in status["reward_history"] if r == 1)
                        accuracy = (correct / total * 100) if total > 0 else 0.0
                        st.metric("Agent Accuracy (%)", f"{accuracy:.1f}%")
                else:
                    st.error("⚠️ Could not fetch agent status.")
            else:
                st.error(f"Error: {resp.text}")
        else:
            st.warning("Candidate ID and feedback text are required.")

# ---------------------------
# Explainability Logs
# ---------------------------
elif page == "Explainability Logs":
    st.header("📊 Explainability Logs")

    candidate_id = None
    if "last_results" in st.session_state:
        candidate_id = st.selectbox("Select Candidate", [c["candidate_id"] for c in st.session_state["last_results"]])
    else:
        candidate_id = st.text_input("Candidate ID to query:", "cv1")

    if st.button("Get Logs"):
        resp = requests.get(f"{API_URL}/agent/explain/{candidate_id}")
        if resp.status_code == 200:
            data = resp.json()
            st.subheader(f"Logs for {data['candidate_id']}")

            for entry in data["timeline"]:
                st.markdown(f"```\n{entry}\n```")

            rewards = []
            for entry in data["timeline"]:
                if "conflicted" in entry:
                    rewards.append(-1)
                elif "matched" in entry:
                    rewards.append(1)
                else:
                    rewards.append(0)
            if rewards:
                st.line_chart(rewards)

            st.download_button(
                "⬇️ Download Logs (JSON)",
                data=json.dumps(data, indent=2),
                file_name=f"{data['candidate_id']}_logs.json",
                mime="application/json"
            )
            df_logs = pd.DataFrame({"timeline": data["timeline"]})
            st.download_button(
                "⬇️ Download Logs (CSV)",
                data=df_logs.to_csv(index=False),
                file_name=f"{data['candidate_id']}_logs.csv",
                mime="text/csv"
            )

            st.success("✅ Logs fetched!")
        else:
            st.error(f"Error: {resp.text}")

# ---------------------------
# Training Monitor
# ---------------------------
elif page == "Training Monitor":
    st.header("📈 RL Agent Training Monitor")

    resp = requests.get(f"{API_URL}/agent/status")
    if resp.status_code == 200:
        data = resp.json()

        st.subheader("Current Weights")
        st.json(data["weights"])

        st.metric("Total Updates", data["total_updates"])
        st.metric("Cumulative Reward", round(data["cumulative_reward"], 2))

        st.subheader("Reward History (last 50)")
        st.line_chart(data["reward_history"])

        if data["reward_history"]:
            # Only consider non-neutral rewards
            total = sum(1 for r in data["reward_history"] if r != 0)
            correct = sum(1 for r in data["reward_history"] if r == 1)
            accuracy = (correct / total * 100) if total > 0 else 0.0
            st.metric("Recent Accuracy (%)", f"{accuracy:.1f}%")
    else:
        st.error(f"Error: {resp.text}")

# ---------------------------
# MDVP Reflections
# ---------------------------
elif page == "MDVP Reflections":
    st.header("🙏 MDVP Reflections (Humility, Gratitude, Integrity)")
    resp = requests.get(f"{API_URL}/agent/mdvp")
    if resp.status_code == 200:
        st.json(resp.json())
    else:
        st.error(f"Error: {resp.text}")

