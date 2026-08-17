import requests
import json
import time
from pathlib import Path

print("=== Audio Pipeline API Test ===")
t0 = time.perf_counter()

# Use jfk.wav as real speech
with open("jfk.wav", "rb") as f:
    wav_bytes = f.read()

print(f"Input WAV size: {len(wav_bytes)} bytes")

r = requests.post(
    "http://127.0.0.1:8000/api/v1/translate-audio",
    files={"audio_file": ("jfk.wav", wav_bytes, "audio/wav")},
    data={"source_language": "en", "target_language": "ur"},
    timeout=300,
)

elapsed = round(time.perf_counter() - t0, 2)
print(f"Total API time: {elapsed}s")
print(f"HTTP Status:    {r.status_code}")

with open("audio_api_result.json", "w", encoding="utf-8") as f:
    if r.headers.get("content-type", "").startswith("application/json"):
        json.dump(r.json(), f, ensure_ascii=False, indent=2)
    else:
        json.dump({"error": r.text[:500]}, f)

d = r.json()
print(f"success:       {d.get('success')}")
print(f"transcription: {d.get('transcription')}")
print(f"translation:   see audio_api_result.json (Urdu text)")
print(f"audio_file:    {d.get('audio_file')}")
print(f"processing_time: {d.get('processing_time')}s")

assert r.status_code == 200, f"FAIL: HTTP {r.status_code}: {d}"
assert d.get("transcription"), "FAIL: transcription is empty"
assert d.get("translation"), "FAIL: translation is empty"
audio_p = Path(d["audio_file"]) if d.get("audio_file") else None
assert audio_p and audio_p.exists(), f"FAIL: audio file not found: {d.get('audio_file')}"
assert audio_p.stat().st_size > 0, "FAIL: audio file is empty"

ext = audio_p.suffix.lower()
mime = "audio/mpeg" if ext == ".mp3" else "audio/wav"
print(f"audio format:  {ext} ({mime})")
print(f"audio size:    {audio_p.stat().st_size} bytes")

print()
print("AUDIO PIPELINE API: PASS")
