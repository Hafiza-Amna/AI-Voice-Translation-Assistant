import requests
import json

# Test 1: Health
r = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=5)
assert r.status_code == 200 and r.json().get("status") == "healthy"
print("Health: PASS (200 OK, status=healthy)")

# Test 2: Text translation (Urdu -> English)
payload = {"text": "\u0622\u067e \u06a9\u06cc\u0633\u06d2 \u06c1\u06cc\u06ba\u061f", "source_language": "ur", "target_language": "en"}
r = requests.post("http://127.0.0.1:8000/api/v1/translate-text", json=payload, timeout=120)
with open("text_translate_result.json", "w", encoding="utf-8") as f:
    json.dump({"status": r.status_code, "json": r.json()}, f, ensure_ascii=False, indent=2)

print(f"Text translate status: {r.status_code}")
d = r.json()
print(f"Success: {d.get('success')}")
print(f"Translation: {d.get('translation')}")
print(f"Audio file: {d.get('audio_file')}")
print(f"Processing time: {d.get('processing_time')}s")

assert r.status_code == 200, f"Expected 200 got {r.status_code}: {d}"
assert d.get("translation"), "Translation field is empty"
print("Text translation: PASS")
