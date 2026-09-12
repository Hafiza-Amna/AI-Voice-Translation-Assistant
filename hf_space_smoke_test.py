"""
Smoke test for hf_space/app.py Gradio 6 messages format.
Tests that all chatbot callbacks produce messages-format dicts
and NOT legacy tuples.
"""
import sys
import os

# Make hf_space importable without actually loading heavy ML models.
# We monkey-patch the heavy imports before importing app logic.
from unittest.mock import MagicMock, patch

hf_stt_mock = MagicMock()
hf_stt_mock.transcribe_audio = MagicMock(return_value="Hello world")

hf_translation_mock = MagicMock()
hf_translation_mock.translate_text = MagicMock(return_value="ہیلو دنیا")

hf_tts_mock = MagicMock()
hf_tts_mock.synthesize_audio = MagicMock(return_value=None)

sys.modules["hf_stt"] = hf_stt_mock
sys.modules["hf_translation"] = hf_translation_mock
sys.modules["hf_tts"] = hf_tts_mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hf_space"))

# Now import the functions from app
from hf_space.app import user_message, bot_response  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
errors = []

def check_messages_format(history, label):
    """Verify every entry in history is a dict with role/content."""
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            errors.append(f"{label} – item[{i}] is {type(msg).__name__}, expected dict")
            return False
        if "role" not in msg or "content" not in msg:
            errors.append(f"{label} – item[{i}] missing 'role'/'content' keys: {msg}")
            return False
        if msg["role"] not in ("user", "assistant"):
            errors.append(f"{label} – item[{i}] has unexpected role: {msg['role']}")
            return False
    return True

# ── Test 1: text EN → UR ─────────────────────────────────────────────────────
history = []
history = user_message("Hello world", None, history)
history = bot_response("Hello world", None, "en", "ur", history)
ok = check_messages_format(history, "text EN→UR")
print(f"[{'PASS' if ok else 'FAIL'}] test 1 – text EN→UR, history length={len(history)}")
if ok:
    print(f"         messages: {[m['role']+': '+str(m['content'])[:40] for m in history]}")

# ── Test 2: text UR → EN ─────────────────────────────────────────────────────
history = []
history = user_message("ہیلو دنیا", None, history)
history = bot_response("ہیلو دنیا", None, "ur", "en", history)
ok = check_messages_format(history, "text UR→EN")
print(f"[{'PASS' if ok else 'FAIL'}] test 2 – text UR→EN, history length={len(history)}")

# ── Test 3: empty input ───────────────────────────────────────────────────────
history = []
history = bot_response("", None, "en", "ur", history)
ok = check_messages_format(history, "empty input")
print(f"[{'PASS' if ok else 'FAIL'}] test 3 – empty input: {history[0]['content'][:40] if history else 'N/A'}")

# ── Test 4: same-language input ───────────────────────────────────────────────
history = []
history = user_message("Test", None, history)
history = bot_response("Test", None, "en", "en", history)
ok = check_messages_format(history, "same-language")
print(f"[{'PASS' if ok else 'FAIL'}] test 4 – same-language error message format OK={ok}")

# ── Test 5: no tuple leakage ─────────────────────────────────────────────────
history = []
history = user_message("Test", None, history)
history = bot_response("Test", None, "en", "ur", history)
tuple_found = any(isinstance(m, tuple) for m in history)
ok = not tuple_found
if not ok:
    errors.append("test 5 – tuple found in history!")
print(f"[{'PASS' if ok else 'FAIL'}] test 5 – no legacy tuples in history")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print("SMOKE TEST FAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All smoke tests passed – Gradio 6 messages format verified.")
    sys.exit(0)
