"""
Smoke test for hf_space/app.py Gradio 6 messages format.
Tests that all chatbot callbacks produce messages-format dicts
and NOT legacy tuples, and that audio output uses gr.Audio for
inline playback (not bare FileData file-download cards).
"""
import sys
import os
import tempfile

import gradio as gr

# Make hf_space importable without actually loading heavy ML models.
from unittest.mock import MagicMock

hf_stt_mock = MagicMock()
hf_stt_mock.transcribe_audio = MagicMock(return_value="Hello world")

hf_translation_mock = MagicMock()
hf_translation_mock.translate_text = MagicMock(return_value="Hello World Translation")

# Create a real temp mp3 file so gr.Audio path validation passes
_tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
_tmp_audio.write(b"\xff\xe0" + b"\x00" * 100)  # minimal fake mp3 bytes
_tmp_audio.close()
FAKE_AUDIO_PATH = _tmp_audio.name

hf_tts_mock = MagicMock()
hf_tts_mock.synthesize_audio = MagicMock(return_value=FAKE_AUDIO_PATH)

sys.modules["hf_stt"] = hf_stt_mock
sys.modules["hf_translation"] = hf_translation_mock
sys.modules["hf_tts"] = hf_tts_mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hf_space"))

from hf_space.app import user_message, bot_response  # noqa: E402

errors = []

def check_messages_format(history, label):
    """Verify every entry is a dict with role/content (not a tuple)."""
    for i, msg in enumerate(history):
        if not isinstance(msg, dict):
            errors.append(f"{label} – item[{i}] is {type(msg).__name__}, expected dict")
            return False
        if "role" not in msg or "content" not in msg:
            errors.append(f"{label} – item[{i}] missing 'role'/'content': {msg}")
            return False
        if msg["role"] not in ("user", "assistant"):
            errors.append(f"{label} – item[{i}] unexpected role: {msg['role']}")
            return False
    return True

def content_type_label(c):
    if isinstance(c, str):
        return f"str({c[:30]})"
    if isinstance(c, gr.Audio):
        return "gr.Audio"
    return type(c).__name__

# ── Test 1: text EN → UR (no audio TTS so TTS returns None) ──────────────────
hf_tts_mock.synthesize_audio = MagicMock(return_value=None)
history = []
history = user_message("Hello world", None, history)
history = bot_response("Hello world", None, "en", "ur", history)
ok = check_messages_format(history, "text EN→UR (no audio)")
print(f"[{'PASS' if ok else 'FAIL'}] test 1 – text EN→UR (no TTS), len={len(history)}")
if ok:
    print(f"         content types: {[content_type_label(m['content']) for m in history]}")

# ── Test 2: text EN → UR WITH audio output ────────────────────────────────────
hf_tts_mock.synthesize_audio = MagicMock(return_value=FAKE_AUDIO_PATH)
history = []
history = user_message("Hello world", None, history)
history = bot_response("Hello world", None, "en", "ur", history)
ok = check_messages_format(history, "text EN→UR+audio")
# Verify the last message is gr.Audio (inline player)
audio_msg = history[-1] if history else {}
is_audio_comp = isinstance(audio_msg.get("content"), gr.Audio)
if not is_audio_comp:
    errors.append(f"test 2 – last message should be gr.Audio, got {content_type_label(audio_msg.get('content'))}")
ok = ok and is_audio_comp
print(f"[{'PASS' if ok else 'FAIL'}] test 2 – text EN→UR with gr.Audio player, len={len(history)}")
print(f"         content types: {[content_type_label(m['content']) for m in history]}")

# ── Test 3: text UR → EN ─────────────────────────────────────────────────────
history = []
history = user_message("ہیلو دنیا", None, history)
history = bot_response("ہیلو دنیا", None, "ur", "en", history)
ok = check_messages_format(history, "text UR→EN")
print(f"[{'PASS' if ok else 'FAIL'}] test 3 – text UR→EN, len={len(history)}")

# ── Test 4: empty input warning ───────────────────────────────────────────────
history = []
history = bot_response("", None, "en", "ur", history)
ok = check_messages_format(history, "empty input")
warning_seen = history and "Please provide text or audio input" in str(history[0].get("content", ""))
if not warning_seen:
    errors.append("test 4 – expected warning for empty input not found")
ok = ok and warning_seen
print(f"[{'PASS' if ok else 'FAIL'}] test 4 – empty input shows warning: {warning_seen}")

# ── Test 5: same-language error ───────────────────────────────────────────────
history = []
history = user_message("Test", None, history)
history = bot_response("Test", None, "en", "en", history)
ok = check_messages_format(history, "same-language")
print(f"[{'PASS' if ok else 'FAIL'}] test 5 – same-language error format OK={ok}")

# ── Test 6: no legacy tuples ──────────────────────────────────────────────────
history = []
history = user_message("Test", None, history)
history = bot_response("Test", None, "en", "ur", history)
tuple_found = any(isinstance(m, tuple) for m in history)
ok = not tuple_found
if not ok:
    errors.append("test 6 – tuple found in history!")
print(f"[{'PASS' if ok else 'FAIL'}] test 6 – no legacy tuples in history")

# ── Test 7: submit flow – text NOT cleared before bot_response ────────────────
# Simulate the event chain: user_message does NOT clear text, so bot_response
# still receives the original text value.
history = []
original_text = "Test submit flow"
history = user_message(original_text, None, history)
# At this point text_input still holds 'original_text' (not yet cleared)
history = bot_response(original_text, None, "en", "ur", history)
ok = check_messages_format(history, "submit flow")
got_warning = any("Please provide text or audio input" in str(m.get("content", "")) for m in history)
if got_warning:
    errors.append("test 7 – false warning appeared in submit flow!")
ok = ok and not got_warning
print(f"[{'PASS' if ok else 'FAIL'}] test 7 – submit flow: text reaches bot_response, no false warning")

# ── Test 8: gr.Audio player has correct value path ────────────────────────────
history = []
history = user_message("Audio path check", None, history)
history = bot_response("Audio path check", None, "en", "ur", history)
audio_entries = [m for m in history if isinstance(m.get("content"), gr.Audio)]
ok = len(audio_entries) == 1
if ok:
    # Verify the Audio component has value set.
    # Gradio may copy the file to its serving cache, so accept:
    # - exact path match, OR
    # - value is a FileData dict whose orig_name or path contains our filename
    audio_val = audio_entries[0]["content"].value
    orig_basename = os.path.basename(FAKE_AUDIO_PATH)
    if isinstance(audio_val, str):
        path_ok = FAKE_AUDIO_PATH in audio_val or orig_basename in audio_val
    elif isinstance(audio_val, dict):
        path_ok = (FAKE_AUDIO_PATH in str(audio_val.get("path", ""))
                   or orig_basename in str(audio_val.get("orig_name", ""))
                   or orig_basename in str(audio_val.get("path", "")))
    else:
        path_ok = False
    if not path_ok:
        errors.append(f"test 8 – gr.Audio value mismatch: {audio_val}")
    ok = ok and path_ok
else:
    errors.append(f"test 8 – expected 1 gr.Audio entry, found {len(audio_entries)}")
print(f"[{'PASS' if ok else 'FAIL'}] test 8 – gr.Audio value path correct")

# ── Cleanup ───────────────────────────────────────────────────────────────────
try:
    os.unlink(FAKE_AUDIO_PATH)
except Exception:
    pass

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if errors:
    print("SMOKE TEST FAILURES:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All smoke tests passed – Gradio 6 inline audio player verified.")
    sys.exit(0)
