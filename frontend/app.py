"""
AI Voice Translation Assistant — ChatGPT-style interface.

A single-page chat UI with:
  • Unified bottom composer:  📎 | [text input] | 🎤 | ➤
  • st.audio_input() for reliable native browser microphone recording
  • Inline file upload (WAV / MP3 / M4A / OGG)
  • Persistent chat history via st.session_state
  • Robust health check and error handling
"""

import html
import os
import time
from pathlib import Path

import httpx
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Voice Translation Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* ── Remove default top padding ── */
    .block-container { padding-top: 0.5rem !important; padding-bottom: 12rem !important; }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.2rem 0 0.6rem;
    }
    .app-title {
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* ── Connection badge ── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 0.22rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-ok  { background: rgba(34,197,94,.15); color:#4ade80; border:1px solid rgba(34,197,94,.3); }
    .badge-err { background: rgba(239,68,68,.15);  color:#f87171; border:1px solid rgba(239,68,68,.3); }

    /* ── Divider ── */
    .chat-divider { border: none; border-top: 1px solid #1e293b; margin: 0.5rem 0 1rem; }

    /* ── Chat bubbles ── */
    .bubble-user {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 18px 18px 4px 18px;
        padding: 0.85rem 1.1rem;
        max-width: 84%;
        margin-left: auto;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.2);
    }
    .bubble-bot {
        background: #0f1e35;
        border: 1px solid rgba(37,99,235,.35);
        border-radius: 18px 18px 18px 4px;
        padding: 1rem 1.2rem;
        max-width: 88%;
        margin-right: auto;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 12px rgba(59,130,246,.08);
    }
    .role-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: 0.35rem;
    }
    .role-user { color: #818cf8; }
    .role-bot  { color: #38bdf8; }
    .ts { color: #475569; font-weight: 400; font-size: 0.7rem; margin-left: 6px; }

    .bubble-text { font-size: 0.97rem; color: #e2e8f0; line-height: 1.55; }

    .chip-label {
        display: inline-flex; align-items: center; gap: 4px;
        font-size: 0.78rem; color: #94a3b8;
        background: rgba(255,255,255,.04);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.2rem 0.6rem;
        margin-bottom: 0.4rem;
    }

    .transcr-block {
        font-size: 0.9rem; color: #cbd5e1;
        background: rgba(255,255,255,.03);
        padding: 0.45rem 0.7rem;
        border-radius: 8px;
        border-left: 3px solid #818cf8;
        margin-bottom: 0.45rem;
    }
    .transl-block {
        font-size: 1rem; font-weight: 600; color: #38bdf8;
        background: rgba(56,189,248,.07);
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 0.45rem;
    }

    /* ── Recording indicator ── */
    .rec-indicator {
        display: flex; align-items: center; gap: 8px;
        background: rgba(239,68,68,.12);
        border: 1px solid rgba(239,68,68,.3);
        border-radius: 12px;
        padding: 0.5rem 1rem;
        font-size: 0.9rem; font-weight: 600; color: #f87171;
    }
    .rec-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #ef4444;
        animation: pulse-rec 1s ease-in-out infinite;
    }
    @keyframes pulse-rec {
        0%,100% { opacity: 1; transform: scale(1); }
        50%      { opacity: .4; transform: scale(.85); }
    }

    /* ── Upload preview chip ── */
    .upload-chip {
        display: flex; align-items: center; gap: 8px;
        background: rgba(129,140,248,.1);
        border: 1px solid rgba(129,140,248,.3);
        border-radius: 10px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem; color: #a5b4fc;
    }

    /* ── Fixed composer ── */
    .composer-outer {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: linear-gradient(to top, #0f172a 70%, transparent);
        padding: 0.6rem 1rem 0.8rem;
        z-index: 9999;
    }
    .composer-inner {
        max-width: 720px;
        margin: 0 auto;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 0.6rem 0.8rem;
        box-shadow: 0 -4px 24px rgba(0,0,0,.35);
    }
    .composer-hint {
        font-size: 0.7rem; color: #475569;
        text-align: center; margin-top: 0.35rem;
    }

    /* hide Streamlit default footer & toolbar */
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { display: none; }
    .stDeployButton { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_URL: str = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

LANGS: dict[str, str] = {"Urdu 🇵🇰": "ur", "English 🇬🇧": "en"}
LANG_LABELS: dict[str, str] = {"ur": "Urdu 🇵🇰", "en": "English 🇬🇧"}

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "conversation_history": [],
    "source_language": "ur",
    "target_language": "en",
    # composer mode: "text" | "mic" | "upload"
    "composer_mode": "text",
    # mic state
    "recorded_audio": None,   # bytes | None — finalised recording
    "recorded_id": None,       # hash to detect new recording
    # upload state
    "pending_upload_bytes": None,
    "pending_upload_name": None,
    "pending_upload_mime": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=6, show_spinner=False)
def _check_health(base_url: str) -> bool:
    """Return True if backend is responding healthy. Cached 6 s to avoid hammering."""
    try:
        r = httpx.get(
            f"{base_url.rstrip('/')}/api/v1/health",
            timeout=4.0,
        )
        if r.status_code == 200:
            ct = r.headers.get("content-type", "")
            if "application/json" in ct:
                data = r.json()
                return isinstance(data, dict) and data.get("status") == "healthy"
    except Exception:
        pass
    return False


def _safe_json(response: httpx.Response) -> dict | None:
    """Return parsed JSON dict only when Content-Type is JSON and parse succeeds."""
    ct = response.headers.get("content-type", "")
    if "application/json" not in ct:
        return None
    try:
        return response.json()
    except Exception:
        return None


def _error_detail(response: httpx.Response) -> str:
    """Extract a human-readable error message from a failed response."""
    data = _safe_json(response)
    if isinstance(data, dict):
        if "detail" in data:
            return str(data["detail"])
        if "message" in data:
            return str(data["message"])
    # Fallback: show status + first 200 chars of body (never raw parse error)
    body_preview = response.text[:200].strip() if response.text else ""
    return f"HTTP {response.status_code}" + (f" — {body_preview}" if body_preview else "")


def _ts() -> str:
    return time.strftime("%I:%M %p")


# ─────────────────────────────────────────────────────────────────────────────
# Send helpers
# ─────────────────────────────────────────────────────────────────────────────

def _send_text(text: str) -> None:
    src = st.session_state["source_language"]
    tgt = st.session_state["target_language"]
    if src == tgt:
        st.error("❌ Source and target languages must differ.")
        return

    # Add user bubble immediately
    st.session_state["conversation_history"].append({
        "role": "user",
        "type": "text",
        "content": text,
        "ts": _ts(),
    })

    url = f"{BACKEND_URL.rstrip('/')}/api/v1/translate-text"
    with st.status("⏳ Translating…", expanded=False) as status_box:
        try:
            r = httpx.post(
                url,
                json={"text": text, "source_language": src, "target_language": tgt},
                timeout=120.0,
            )
        except httpx.ConnectError:
            status_box.update(label="❌ Connection failed", state="error")
            st.session_state["conversation_history"].pop()
            st.error("❌ Cannot reach backend — is FastAPI running on port 8000?")
            return
        except Exception as exc:
            status_box.update(label="❌ Unexpected error", state="error")
            st.session_state["conversation_history"].pop()
            st.error(f"❌ {exc}")
            return

        if r.status_code != 200:
            status_box.update(label=f"❌ Translation failed", state="error")
            st.session_state["conversation_history"].pop()
            st.error(f"❌ Translation failed: {_error_detail(r)}")
            return

        data = _safe_json(r)
        if data is None:
            status_box.update(label="❌ Bad server response", state="error")
            st.session_state["conversation_history"].pop()
            st.error("❌ Server returned a non-JSON response.")
            return

        status_box.update(label="✅ Done!", state="complete")

    # Read audio bytes from the path returned by the server
    audio_bytes: bytes | None = None
    audio_path = data.get("audio_file", "")
    if audio_path:
        p = Path(audio_path)
        if p.exists():
            try:
                audio_bytes = p.read_bytes()
            except Exception:
                audio_bytes = None

    st.session_state["conversation_history"].append({
        "role": "assistant",
        "type": "text",
        "original": text,
        "translation": data.get("translation", ""),
        "audio": audio_bytes,
        "proc": data.get("processing_time", 0.0),
        "ts": _ts(),
    })
    st.rerun()


def _send_audio(audio_bytes: bytes, filename: str, mime: str) -> None:
    src = st.session_state["source_language"]
    tgt = st.session_state["target_language"]
    if src == tgt:
        st.error("❌ Source and target languages must differ.")
        return

    # Add user bubble
    st.session_state["conversation_history"].append({
        "role": "user",
        "type": "audio",
        "content": f"🎙️ Voice message ({filename})",
        "audio": audio_bytes,
        "ts": _ts(),
    })

    url = f"{BACKEND_URL.rstrip('/')}/api/v1/translate-audio"
    with st.status("⏳ Processing audio…", expanded=True) as status_box:
        status_box.write("1️⃣  Transcribing speech…")
        try:
            r = httpx.post(
                url,
                files={"audio_file": (filename, audio_bytes, mime)},
                data={"source_language": src, "target_language": tgt},
                timeout=180.0,
            )
        except httpx.ConnectError:
            status_box.update(label="❌ Connection failed", state="error")
            st.session_state["conversation_history"].pop()
            st.error("❌ Cannot reach backend — is FastAPI running on port 8000?")
            return
        except Exception as exc:
            status_box.update(label="❌ Unexpected error", state="error")
            st.session_state["conversation_history"].pop()
            st.error(f"❌ {exc}")
            return

        if r.status_code != 200:
            detail = _error_detail(r)
            status_box.update(label="❌ Processing failed", state="error")
            st.session_state["conversation_history"].pop()
            if "speech-to-text" in detail.lower() or "transcription" in detail.lower():
                st.error(f"❌ Speech recognition failed: {detail}")
            elif "translation" in detail.lower():
                st.error(f"❌ Translation failed: {detail}")
            elif "speech synthesis" in detail.lower() or "tts" in detail.lower():
                st.error(f"❌ Text-to-speech failed: {detail}")
            else:
                st.error(f"❌ Backend error: {detail}")
            return

        data = _safe_json(r)
        if data is None:
            status_box.update(label="❌ Bad server response", state="error")
            st.session_state["conversation_history"].pop()
            st.error("❌ Server returned a non-JSON response.")
            return

        status_box.update(label="✅ Done!", state="complete")

    # Read TTS audio bytes
    audio_out: bytes | None = None
    audio_fmt: str = "audio/wav"
    audio_path = data.get("audio_file", "")
    if audio_path:
        p = Path(audio_path)
        if p.exists():
            try:
                audio_out = p.read_bytes()
                # Detect format from extension (gTTS produces .mp3, Piper .wav)
                ext = p.suffix.lower()
                audio_fmt = "audio/mpeg" if ext == ".mp3" else "audio/wav"
            except Exception:
                audio_out = None

    st.session_state["conversation_history"].append({
        "role": "assistant",
        "type": "audio",
        "original": data.get("transcription", ""),
        "translation": data.get("translation", ""),
        "audio": audio_out,
        "audio_fmt": audio_fmt,
        "proc": data.get("processing_time", 0.0),
        "ts": _ts(),
    })

    # Clear pending audio state
    st.session_state["recorded_audio"] = None
    st.session_state["recorded_id"] = None
    st.session_state["pending_upload_bytes"] = None
    st.session_state["pending_upload_name"] = None
    st.session_state["composer_mode"] = "text"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    if st.button("🔄 Refresh connection", use_container_width=True):
        _check_health.clear()
        st.rerun()
    st.markdown("---")
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state["conversation_history"] = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Backend URL**")
    st.code(BACKEND_URL, language=None)
    st.markdown("**API Docs**")
    st.markdown(f"[Swagger UI]({BACKEND_URL}/docs)")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
healthy = _check_health(BACKEND_URL)
badge_html = (
    "<span class='badge badge-ok'>🟢 Connected</span>"
    if healthy
    else "<span class='badge badge-err'>🔴 Disconnected</span>"
)
st.markdown(
    f"""
    <div class='app-header'>
      <div class='app-title'>🎙️ AI Voice Translation Assistant</div>
      <div class='app-subtitle'>Speak or type — get instant Urdu ↔ English translation</div>
      <div style='margin-top:0.5rem'>{badge_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not healthy:
    st.warning(
        "⚠️ Backend is not reachable. "
        "Make sure `uvicorn app.main:app --port 8000` is running, "
        "then click **Refresh connection** in the sidebar."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Language row
# ─────────────────────────────────────────────────────────────────────────────
lc1, _swap, lc2 = st.columns([5, 1, 5])
with lc1:
    src_lbl = st.selectbox(
        "Source",
        list(LANGS.keys()),
        index=0 if st.session_state["source_language"] == "ur" else 1,
        key="sel_src",
        label_visibility="visible",
    )
    st.session_state["source_language"] = LANGS[src_lbl]
with _swap:
    st.markdown("<div style='text-align:center;padding-top:1.8rem;font-size:1.3rem;color:#64748b'>⇄</div>", unsafe_allow_html=True)
with lc2:
    tgt_lbl = st.selectbox(
        "Target",
        list(LANGS.keys()),
        index=1 if st.session_state["target_language"] == "en" else 0,
        key="sel_tgt",
        label_visibility="visible",
    )
    st.session_state["target_language"] = LANGS[tgt_lbl]

st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat History
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state["conversation_history"]:
    st.markdown(
        """
        <div style='text-align:center;padding:3rem 1rem;color:#475569'>
          <div style='font-size:2.5rem;margin-bottom:0.5rem'>💬</div>
          <div style='font-size:1rem;font-weight:600;color:#64748b'>Start a conversation</div>
          <div style='font-size:0.85rem;margin-top:0.3rem'>Type below, record your voice, or upload an audio file</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for item in st.session_state["conversation_history"]:
        if item["role"] == "user":
            content_html = html.escape(str(item.get("content", "")))
            st.markdown(
                f"""<div class='bubble-user'>
                  <div class='role-label role-user'>👤 You<span class='ts'>{html.escape(item.get('ts', ''))}</span></div>
                  <div class='bubble-text'>{content_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if item.get("audio"):
                st.audio(item["audio"], format="audio/wav")

        else:  # assistant
            original_html = html.escape(str(item.get("original", "")))
            translation_html = html.escape(str(item.get("translation", "")))
            proc = item.get("proc", 0.0)
            item_type = item.get("type", "text")

            if item_type == "audio":
                orig_label = "📝 <b>Transcription (original):</b>"
            else:
                orig_label = "📝 <b>Original text:</b>"

            st.markdown(
                f"""<div class='bubble-bot'>
                  <div class='role-label role-bot'>🤖 Assistant<span class='ts'>⚡ {proc:.2f}s</span></div>
                  <div class='transcr-block'>{orig_label} {original_html}</div>
                  <div class='transl-block'>🌐 <b>Translation:</b> {translation_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if item.get("audio"):
                st.markdown("🔊 **Translated voice:**")
                st.audio(item["audio"], format=item.get("audio_fmt", "audio/wav"))


# ─────────────────────────────────────────────────────────────────────────────
# Composer — fixed at bottom
# ─────────────────────────────────────────────────────────────────────────────
# We render the composer area using a container. The CSS positions it fixed.
# Streamlit doesn't support true fixed widgets, so we use a sticky bottom
# approach: the chat area has bottom padding = composer height.

st.markdown("<div class='composer-outer'><div class='composer-inner'>", unsafe_allow_html=True)

# ── Row 1: pending state (upload preview / recording state) ──────────────────
mode = st.session_state["composer_mode"]

if mode == "mic":
    # ── MICROPHONE RECORDING MODE ──────────────────────────────────────────
    st.markdown(
        "<div class='rec-indicator'><div class='rec-dot'></div>"
        " Recording… click the microphone widget to start, then stop when done.</div>",
        unsafe_allow_html=True,
    )
    recorded = st.audio_input(
        "🎤 Tap to start recording — tap again to stop",
        label_visibility="visible",
        key="mic_input",
    )
    if recorded is not None:
        raw = recorded.getvalue()
        rid = hash(raw)
        if rid != st.session_state["recorded_id"]:
            st.session_state["recorded_audio"] = raw
            st.session_state["recorded_id"] = rid

    if st.session_state["recorded_audio"]:
        st.markdown("✅ **Recording captured** — preview & send:")
        st.audio(st.session_state["recorded_audio"], format="audio/wav")
        col_send, col_discard = st.columns([3, 1])
        with col_send:
            if st.button("➤ Send Voice", type="primary", use_container_width=True, key="btn_send_mic"):
                _send_audio(
                    audio_bytes=st.session_state["recorded_audio"],
                    filename="recording.wav",
                    mime="audio/wav",
                )
        with col_discard:
            if st.button("🗑 Discard", use_container_width=True, key="btn_discard_mic"):
                st.session_state["recorded_audio"] = None
                st.session_state["recorded_id"] = None
                st.session_state["composer_mode"] = "text"
                st.rerun()
    else:
        col_cancel, = st.columns([1])
        if st.button("✖ Cancel Recording", use_container_width=True, key="btn_cancel_mic"):
            st.session_state["composer_mode"] = "text"
            st.rerun()

elif mode == "upload":
    # ── FILE UPLOAD MODE ────────────────────────────────────────────────────
    up_file = st.file_uploader(
        "Choose audio file (.wav .mp3 .m4a .ogg)",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="visible",
        key="file_uploader",
    )
    if up_file is not None:
        if st.session_state["pending_upload_name"] != up_file.name:
            st.session_state["pending_upload_bytes"] = up_file.getvalue()
            st.session_state["pending_upload_name"] = up_file.name
            st.session_state["pending_upload_mime"] = up_file.type or "audio/wav"

    if st.session_state["pending_upload_bytes"]:
        st.markdown(
            f"<div class='upload-chip'>📎 <b>{html.escape(st.session_state['pending_upload_name'])}</b></div>",
            unsafe_allow_html=True,
        )
        st.audio(st.session_state["pending_upload_bytes"], format=st.session_state["pending_upload_mime"])
        col_send_u, col_discard_u = st.columns([3, 1])
        with col_send_u:
            if st.button("➤ Send Audio File", type="primary", use_container_width=True, key="btn_send_upload"):
                _send_audio(
                    audio_bytes=st.session_state["pending_upload_bytes"],
                    filename=st.session_state["pending_upload_name"],
                    mime=st.session_state["pending_upload_mime"],
                )
        with col_discard_u:
            if st.button("🗑 Clear", use_container_width=True, key="btn_clear_upload"):
                st.session_state["pending_upload_bytes"] = None
                st.session_state["pending_upload_name"] = None
                st.session_state["composer_mode"] = "text"
                st.rerun()
    else:
        if st.button("✖ Cancel Upload", use_container_width=True, key="btn_cancel_upload"):
            st.session_state["composer_mode"] = "text"
            st.rerun()

else:
    # ── TEXT / DEFAULT MODE ─────────────────────────────────────────────────
    # Composer row: 📎 | text input | 🎤 | ➤
    col_att, col_txt, col_mic, col_send = st.columns([1, 10, 1, 1])

    with col_att:
        if st.button("📎", help="Upload audio file", key="btn_att", use_container_width=True):
            st.session_state["composer_mode"] = "upload"
            st.rerun()

    with col_txt:
        text_val = st.text_input(
            "message",
            placeholder="Type a message in Urdu or English…",
            label_visibility="collapsed",
            key="text_msg",
        )

    with col_mic:
        if st.button("🎤", help="Record voice message", key="btn_mic", use_container_width=True):
            st.session_state["composer_mode"] = "mic"
            st.rerun()

    with col_send:
        send_clicked = st.button("➤", type="primary", key="btn_send_text", use_container_width=True)

    if send_clicked:
        if text_val and text_val.strip():
            _send_text(text_val.strip())
        else:
            st.warning("⚠️ Type a message first, or use 🎤 / 📎 to send audio.")

    st.markdown(
        "<div class='composer-hint'>➤ Send text · 🎤 Record voice · 📎 Upload audio file</div>",
        unsafe_allow_html=True,
    )

st.markdown("</div></div>", unsafe_allow_html=True)
