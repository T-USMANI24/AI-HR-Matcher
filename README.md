# AI-HR-Matcher
 CV parsing, JD analysis, skill &amp; degree matching, and automated candidate evaluation with RL-based decision support.
 An AI-powered system that matches Job Descriptions (JDs) with Candidate CVs, collects recruiter feedback, and adapts using Reinforcement Learning (RL).

🚀 Built with Python, Streamlit, FlaskAPI, SBERT, and TF-IDF.
📊 Provides transparency via Explainability Logs, Training Monitor, and Reward Curves.
🙏 Embeds MDVP values (Humility, Gratitude, Integrity) for ethical AI.

✨ Features

Upload JDs and CVs (TXT/PDF) for automatic similarity scoring.

Multiple similarity methods: TF-IDF and Sentence-BERT.

Reinforcement Learning Agent

Adapts its weights based on recruiter feedback.

Tracks cumulative reward, updates, and accuracy.

Feedback Integration

Recruiter provides feedback + values like integrity, honesty, discipline, hard work.

Explainability Logs

Shows decision timeline, rewards, and conflicts.

Training Monitor

Displays agent weights, cumulative rewards, and reward history charts.

MDVP Reflections

Logs agent’s growth in humility, gratitude, and integrity.

🖼️ Demo Screens

Candidate Matching

Feedback Submission with Reward Curve

Explainability Logs

Training Monitor with Reward Chart

(Add screenshots/gifs here later)

⚙️ Tech Stack

Frontend: Streamlit

Backend: FlaskAPI

NLP Models:

TF-IDF (scikit-learn)

Sentence-BERT (transformers)

RL Agent: Custom reward-based learning agent (Python, NumPy)

Storage: In-memory (can extend to DB later)

Explainability: Custom logs + visualizations

## 📂 Project Structure

📦 Candidate-Matching-Agent
 Candidate-Matching-Agent/
'''│── app.py # Streamlit frontend
│── agent.py # RL agent core logic
│── backend.py # FastAPI backend server
│── requirements.txt # Python dependencies
│── README.md # Documentation
│
├── data/ # Sample CVs and JDs
│ ├── sample_jd.txt
│ ├── sample_cv1.txt
│ └── sample_cv2.txt
│
├── logs/ # Explainability & feedback logs
│ └── agent_logs.json
│
└── Dockerfile'''

1️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Backend (FlaskAPI)
python agent.py

4️⃣Run Frontend (Streamlit)
streamlit run app.py

🧑‍💻 Usage Flow

Go to Match Candidates → Upload JD + CVs → Get similarity scores.

Go to Submit Feedback → Provide recruiter feedback + values.

RL agent updates weights → Reward curve is updated.

Training Monitor shows agent’s learning progress.

Explainability Logs provide transparency of decisions.

MDVP Reflections highlight ethical AI learning.

📈 Example Reward Curve

The RL agent learns from recruiter corrections:

Positive feedback → Reward +1

Negative feedback → Reward -1

Neutral/no correction → Reward 0

This helps improve candidate matching accuracy over time.

🔮 Future Improvements

Integrate database (Postgres/MongoDB) for persistence.

Deploy as a cloud-based web app (AWS/GCP/Heroku).

Add more NLP models like GPT embeddings.

Improve explainability with SHAP/LIME.

Multi-user role support (Recruiter, Admin, Candidate).

🙏 Values – MDVP

This project emphasizes:

Humility – AI learns from mistakes.

Discipline – Consistent reward-driven learning.

Gratitude – Thanks to recruiter feedback.

Integrity – Transparent decision logs.

👨‍💻 Author

Developed by Talha usmani
💼 For research, hiring, and fair AI decision-making.
