import requests
import json

# Candidate ID you want to inspect (use "cv1" or "cv2" from test_match.py)
candidate_id = "cv1"

url = f"http://localhost:8000/agent/explain/{candidate_id}"

resp = requests.get(url)
print("Status:", resp.status_code)

if resp.status_code == 200:
    data = resp.json()
    print(json.dumps(data, indent=2))
else:
    print("Error:", resp.text)
