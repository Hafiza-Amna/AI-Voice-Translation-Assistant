import os
import gradio as gr
from hf_stt import transcribe_audio
from hf_translation import translate_text
from hf_tts import synthesize_audio


def process_audio(audio_path, source_lang, target_lang):
    """Full pipeline: audio file → STT → Translation → TTS."""
    if not audio_path:
        raise gr.Error("Please record or upload an audio file first.")
    if source_lang == target_lang:
        raise gr.Error("Source and target languages must be different.")

    transcription = transcribe_audio(audio_path)
    if not transcription:
        raise gr.Error("No speech detected in the audio. Please try again.")

    translation = translate_text(transcription, source_lang, target_lang)
    audio_out = synthesize_audio(translation, target_lang)

    return transcription, translation, audio_out


def process_text(text, source_lang, target_lang):
    """Text-only pipeline: text → Translation → TTS."""
    if not text or not text.strip():
        raise gr.Error("Please type a message first.")
    if source_lang == target_lang:
        raise gr.Error("Source and target languages must be different.")

    translation = translate_text(text, source_lang, target_lang)
    audio_out = synthesize_audio(translation, target_lang)

    return translation, audio_out


# ── Custom CSS ─────────────────────────────────────────────────────────────
custom_css = """
/* App background and font */
body {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Header styling */
.app-title {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center;
}
.app-subtitle {
    font-size: 0.88rem !important;
    color: #94a3b8 !important;
    text-align: center;
    margin-top: 0.2rem !important;
    margin-bottom: 1rem !important;
}

/* Chatbot container */
#chatbot {
    background: #0f172a !important;
    border: none !important;
    box-shadow: none !important;
}

/* Message bubbles */
.user-row {
    justify-content: flex-end;
}
.bot-row {
    justify-content: flex-start;
}
.message.user {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #e2e8f0 !important;
}
.message.bot {
    background: #0f1e35 !important;
    border: 1px solid rgba(37,99,235,.35) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: #e2e8f0 !important;
}

/* Controls container at the bottom */
#controls-container {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    padding: 0.6rem 0.8rem !important;
    box-shadow: 0 -4px 24px rgba(0,0,0,.35) !important;
}

/* Audio component styling */
audio {
    border-radius: 8px !important;
    height: 40px !important;
}
"""

def user_message(text, audio_path, history):
    if audio_path:
        # Show audio in history
        history.append(((audio_path,), None))
    elif text:
        history.append((text, None))
    return gr.update(value="", interactive=False), history

def bot_response(text, audio_path, source_lang, target_lang, history):
    if not text and not audio_path:
        history.append((None, "⚠️ Please provide text or audio input."))
        return history
    
    if source_lang == target_lang:
        history.append((None, "❌ Source and target languages must be different."))
        return history

    try:
        if audio_path:
            # Full Pipeline
            transcription = transcribe_audio(audio_path)
            if not transcription:
                history.append((None, "❌ No speech detected in the audio. Please try again."))
                return history
            
            translation = translate_text(transcription, source_lang, target_lang)
            audio_out = synthesize_audio(translation, target_lang)
            
            response_md = f"📝 **Transcription:** {transcription}\n\n🌐 **Translation:** {translation}"
            history.append((None, response_md))
            if audio_out:
                history.append((None, (audio_out,)))
        else:
            # Text Pipeline
            translation = translate_text(text, source_lang, target_lang)
            audio_out = synthesize_audio(translation, target_lang)
            
            response_md = f"🌐 **Translation:** {translation}"
            history.append((None, response_md))
            if audio_out:
                history.append((None, (audio_out,)))
                
    except Exception as e:
        history.append((None, f"❌ Error: {str(e)}"))
        
    return history


# ── Gradio UI ──────────────────────────────────────────────────────────────
with gr.Blocks(title="AI Voice Translation Assistant") as demo:
    gr.HTML("<div class='app-title'>🎙️ AI Voice Translation Assistant</div>")
    gr.HTML("<div class='app-subtitle'>Speak or type — get instant Urdu ↔ English translation powered by Whisper + NLLB-200.</div>")

    with gr.Row():
        source_lang = gr.Dropdown(
            choices=[("Urdu 🇵🇰", "ur"), ("English 🇬🇧", "en")],
            value="ur",
            label="Source Language",
            scale=1
        )
        gr.HTML("<div style='text-align:center;padding-top:2rem;font-size:1.3rem;color:#64748b'>⇄</div>", scale=0)
        target_lang = gr.Dropdown(
            choices=[("Urdu 🇵🇰", "ur"), ("English 🇬🇧", "en")],
            value="en",
            label="Target Language",
            scale=1
        )

    chatbot = gr.Chatbot(
        label="Conversation",
        elem_id="chatbot",
        height=500,
    )

    with gr.Column(elem_id="controls-container"):
        with gr.Row():
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Type a message in Urdu or English...",
                container=False,
                scale=8
            )
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                scale=2,
                show_label=False
            )
            submit_btn = gr.Button("➤ Send", variant="primary", scale=1)

    # Event handlers
    text_input.submit(
        fn=user_message,
        inputs=[text_input, audio_input, chatbot],
        outputs=[text_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=[text_input, audio_input, source_lang, target_lang, chatbot],
        outputs=chatbot,
    ).then(
        fn=lambda: None,
        inputs=None,
        outputs=audio_input, # Clear audio after send
        queue=False
    )
    
    submit_btn.click(
        fn=user_message,
        inputs=[text_input, audio_input, chatbot],
        outputs=[text_input, chatbot],
        queue=False
    ).then(
        fn=bot_response,
        inputs=[text_input, audio_input, source_lang, target_lang, chatbot],
        outputs=chatbot,
    ).then(
        fn=lambda: None,
        inputs=None,
        outputs=audio_input, # Clear audio after send
        queue=False
    )

if __name__ == "__main__":
    demo.queue().launch(share=False, css=custom_css)

