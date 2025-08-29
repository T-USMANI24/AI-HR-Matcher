import requests
import json

url = "http://localhost:8000/agent/match"

with open("data/sample_jds/jd1.txt", "r") as f:
    jd_text = f.read()

cvs = []
for i in [1, 2]:
    with open(f"data/sample_cvs/cv{i}.txt", "r") as f:
        cvs.append({
            "id": f"cv{i}",
            "text": f.read(),
            "values": {"integrity": 4, "honesty": 5, "discipline": 3, "hard_work": 4}
        })

payload = {
    "jd_text": jd_text,
    "cvs": cvs,
    "method": "tfidf"   # or "sbert" if you installed sentence-transformers
}

resp = requests.post(url, json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
