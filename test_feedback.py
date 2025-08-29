import requests
import json

url = "http://localhost:8000/agent/feedback"

payload = {
    "candidate_id": "cv1",
    "feedback_text": "Very strong technical skills and confident communication.",
    "values": {"integrity": 5, "honesty": 5, "discipline": 4, "hard_work": 5},
    "original_match": 0.75
}

resp = requests.post(url, json=payload)
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
