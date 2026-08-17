"""
End-to-end pipeline verification script.
Tests STT -> Translation -> TTS with a real WAV file.
Run: python e2e_verify.py
"""
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

results = {}

# ── 1. STT ────────────────────────────────────────────────────────────────────
section("Stage 1: STT (faster-whisper)")
from app.services.stt_service import STTService

wav_path = "jfk.wav"
assert Path(wav_path).exists(), f"Test WAV missing: {wav_path}"
print(f"Input WAV: {wav_path}  size={Path(wav_path).stat().st_size} bytes")

stt = STTService()
t0 = time.perf_counter()
stt_result = stt.transcribe(audio_path=wav_path)
stt_time = round(time.perf_counter() - t0, 2)

print(f"STT TEXT:     '{stt_result.text}'")
print(f"STT LANGUAGE: {stt_result.language}")
print(f"STT TIME:     {stt_time}s")

assert stt_result.text.strip(), "FAIL: STT returned empty text"
results["stt"] = {"pass": True, "text": stt_result.text, "time": stt_time}
print("STT: PASS")

# ── 2. Translation ───────────────────────────────────────────────────────────
section("Stage 2: NLLB Translation")
from app.services.translation_service import TranslationService

ts = TranslationService()
t0 = time.perf_counter()
trans_result = ts.translate(text=stt_result.text, source_lang="en", target_lang="ur")
trans_time = round(time.perf_counter() - t0, 2)

print(f"TRANSLATION INPUT:  '{trans_result.original_text}'")
print(f"TRANSLATION OUTPUT: see e2e_results.json (Urdu)")
print(f"TRANSLATION TIME:   {trans_time}s")

assert trans_result.translated_text.strip(), "FAIL: Translation returned empty text"
results["translation"] = {"pass": True, "text": trans_result.translated_text, "time": trans_time}
print("Translation: PASS")

# ── 3. TTS ───────────────────────────────────────────────────────────────────
section("Stage 3: TTS")
from app.services.tts_service import TTSService

tts = TTSService()
t0 = time.perf_counter()
tts_result = tts.synthesize(text=trans_result.translated_text, language="ur")
tts_time = round(time.perf_counter() - t0, 2)

audio_path = Path(tts_result.audio_path)
audio_size = audio_path.stat().st_size if audio_path.exists() else 0
audio_ext = audio_path.suffix.lower()
mime = "audio/mpeg" if audio_ext == ".mp3" else "audio/wav"

print(f"TTS OUTPUT PATH:  {tts_result.audio_path}")
print(f"TTS FORMAT:       {audio_ext} ({mime})")
print(f"TTS FILE SIZE:    {audio_size} bytes")
print(f"TTS TIME:         {tts_time}s")

assert audio_path.exists(), f"FAIL: TTS output file does not exist: {tts_result.audio_path}"
assert audio_size > 0, f"FAIL: TTS output file is empty: {tts_result.audio_path}"
results["tts"] = {"pass": True, "path": str(audio_path), "mime": mime, "size": audio_size, "time": tts_time}
print("TTS: PASS")

# ── 4. Summary ───────────────────────────────────────────────────────────────
section("Summary")
total = stt_time + trans_time + tts_time
print(f"STT:         {stt_time}s  <- {'*** SLOW ***' if stt_time > 30 else 'OK'}")
print(f"Translation: {trans_time}s  <- {'*** SLOW ***' if trans_time > 30 else 'OK'}")
print(f"TTS:         {tts_time}s  <- {'*** SLOW ***' if tts_time > 30 else 'OK'}")
print(f"TOTAL:       {total}s")
print()
print(f"Audio file: {tts_result.audio_path}")
print(f"Audio size: {audio_size} bytes")
print(f"Audio MIME: {mime}")
print()

# Write JSON output
with open("e2e_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "stt_text": stt_result.text,
        "translation_text": trans_result.translated_text,
        "audio_path": str(audio_path),
        "audio_mime": mime,
        "audio_size": audio_size,
        "times": {"stt": stt_time, "translation": trans_time, "tts": tts_time, "total": total},
    }, f, ensure_ascii=False, indent=2)

print("Results written to e2e_results.json")
print("ALL STAGES PASSED")
