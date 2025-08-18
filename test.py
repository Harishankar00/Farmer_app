import time
import requests

def call_gemini(payload, retries=3):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=YOUR_API_KEY"
    for attempt in range(retries):
        resp = requests.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            wait = (2 ** attempt)  # exponential backoff
            print(f"⏳ Rate limited, retrying in {wait}s...")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    raise Exception("Failed after retries")
