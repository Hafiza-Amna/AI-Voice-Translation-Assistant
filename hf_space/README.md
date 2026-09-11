---
title: AI Voice Translation Assistant
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.27.0"
app_file: app.py
pinned: false
license: mit
short_description: Urdu ↔ English voice & text translation using Whisper + NLLB-200
tags:
  - speech-to-text
  - translation
  - text-to-speech
  - urdu
  - english
  - whisper
  - nllb
---

# 🎙️ AI Voice Translation Assistant

Speak or type — get instant **Urdu ↔ English** translation powered by:

- **STT**: OpenAI Whisper (small) via `transformers`
- **Translation**: Facebook NLLB-200-distilled-600M
- **TTS**: gTTS (Google Text-to-Speech)

## Usage

1. Select **Source Language** and **Target Language** from the dropdowns.
2. Either **type a message** in the text box or **record / upload audio** using the microphone button.
3. Click **➤ Send** to translate.
4. The translated text and synthesised audio will appear in the chat window.

## Hardware

This Space uses **ZeroGPU** — GPU resources are allocated dynamically per request.
