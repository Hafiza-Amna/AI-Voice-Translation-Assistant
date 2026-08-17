# Project Documentation: AI Voice Translation Assistant

## 1. Project Overview
The AI Voice Translation Assistant is a comprehensive, offline-first machine learning application designed to bridge the language gap between Urdu and English. The system accepts spoken audio, uploaded audio files, or text, and seamlessly translates it into the target language. The translated text is then synthesized back into human-like speech. It operates entirely locally using robust pretrained AI models, ensuring data privacy and removing reliance on paid third-party cloud APIs.

## 2. Problem Statement
Language barriers remain a significant hurdle in global communication, particularly for low-resource languages like Urdu. Existing translation solutions often lack bidirectional voice support, require constant internet connectivity, or rely on costly third-party APIs that compromise user privacy. There is a critical need for an accessible, offline, and privacy-focused system capable of instantly processing spoken and written Urdu and English.

## 3. Objectives
- **End-to-End Voice Translation:** Enable users to speak in one language and hear the translation in another.
- **Offline Capability:** Utilize local pretrained models to perform Speech-to-Text (STT), Machine Translation, and Text-to-Speech (TTS) without internet access.
- **User-Friendly Interface:** Provide a ChatGPT-style conversational web UI for easy interaction.
- **Robust Architecture:** Develop a decoupled backend and frontend to ensure scalability, ease of testing, and maintainability.
- **Privacy & Security:** Ensure all audio and text data remains strictly on the host machine.

## 4. Features Implemented
- **Speech-to-Text (STT):** High-accuracy offline transcription of spoken audio.
- **Bidirectional Machine Translation:** Seamless translation between Urdu and English.
- **Text-to-Speech (TTS):** Neural voice synthesis of the translated text.
- **Multimodal Input:** Support for live microphone recording, audio file uploads (`.wav`, `.mp3`, `.m4a`, `.ogg`), and raw text input.
- **Conversational UI:** Streamlit-based web interface displaying original text, translations, and interactive audio players.
- **RESTful API Backend:** High-performance FastAPI server managing the inference pipeline.
- **Health Monitoring:** API endpoints to verify system readiness and backend connectivity.

## 5. Technology Stack
- **Programming Language:** Python 3.12+
- **Frontend Framework:** Streamlit
- **Backend Framework:** FastAPI, Uvicorn
- **Machine Learning & AI Libraries:**
  - `faster-whisper` (CTranslate2) for STT.
  - `transformers` (Hugging Face) for Machine Translation.
  - `gTTS` / `Piper` for TTS.
- **Testing Framework:** Pytest, `httpx` (for async API testing)
- **Environment Management:** Python `venv`, `.env` configuration (via Pydantic BaseSettings).

## 6. Project Architecture / Workflow
The system utilizes a decoupled, client-server architecture:

1. **User Input:** The user interacts with the Streamlit frontend via text input, microphone recording, or file upload.
2. **Frontend Transmission:** The Streamlit app sends a multipart/form-data HTTP POST request to the FastAPI backend (`/api/v1/translate-audio` or `/api/v1/translate-text`).
3. **Backend Processing (FastAPI):**
   - **Validation:** Validates file size, format, and language parameters.
   - **Speech-to-Text:** Audio is passed to the `STTService` (powered by `faster-whisper`) to extract the transcript.
   - **Translation:** The transcript is passed to the `TranslationService` (powered by NLLB-200) to generate the target language text.
   - **Text-to-Speech:** The translated text is passed to the `TTSService` (powered by `gTTS`/`Piper`) to generate an MP3/WAV file.
4. **Response:** The backend returns a JSON payload containing the transcription, translation, processing time, and the path to the generated audio file.
5. **UI Rendering:** The frontend displays the text and renders an HTML5 audio player for the generated speech.

## 7. Folder Structure
```text
AI-Voice-Translation-Assistant/
├── app/                            # Backend FastAPI source code
│   ├── api/                        # API routes and dependencies
│   ├── models/                     # Pydantic schemas (Request/Response)
│   ├── services/                   # AI integration logic (STT, TTS, Translation)
│   ├── utils/                      # Helper scripts (audio validation, etc.)
│   ├── config.py                   # Environment configuration loader
│   └── main.py                     # Application entry point
├── frontend/                       # Streamlit Web UI source code
│   ├── components/                 # Reusable UI widgets
│   └── app.py                      # Main Streamlit application
├── tests/                          # Automated Test Suite (Pytest)
│   ├── integration/                # End-to-end API tests
│   └── unit/                       # Component-level tests
├── screenshots/                    # UI screenshots for documentation
├── audio_temp/                     # Temporary directory for audio uploads
├── generated_audio/                # Temporary directory for TTS output
├── requirements.txt                # Python package dependencies
├── .env.example                    # Template for environment variables
└── README.md / Documentation.md    # Project documentation files
```

## 8. Installation and Setup Guide
1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd AI-Voice-Translation-Assistant
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   ```bash
   copy .env.example .env
   # Edit .env with your specific paths if necessary.
   ```

## 9. Environment Requirements
- **OS:** Windows, macOS, or Linux.
- **Python:** Version 3.12 or higher.
- **FFmpeg:** Must be installed and added to the system PATH (required by faster-whisper for audio processing).
- **RAM:** Minimum 4GB (8GB recommended) to hold the models in memory.

## 10. How to Run the Project

**Start the FastAPI Backend:**
Open a terminal, ensure the virtual environment is activated, and run:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
*Note: The API docs will be available at `http://127.0.0.1:8000/docs`.*

**Start the Streamlit Frontend:**
Open a new terminal, activate the virtual environment, and run:
```bash
streamlit run frontend/app.py
```
*The web interface will open at `http://localhost:8501`.*

## 11. Model Architecture / Algorithm Explanation
- **Speech-to-Text (STT):** Uses `faster-whisper`, a reimplementation of OpenAI's Whisper model using CTranslate2. It applies a Transformer-based sequence-to-sequence architecture to robustly transcribe audio.
- **Machine Translation:** Employs Meta's `NLLB-200-distilled-600M` (No Language Left Behind). This is an encoder-decoder Transformer model distilled to 600 million parameters, specifically optimized for low-resource languages like Urdu (`urd_Arab`).
- **Text-to-Speech (TTS):** Uses a fallback mechanism. If the `Piper` neural TTS engine is unavailable, the system automatically falls back to `gTTS` (Google TTS) which synthesizes high-quality MP3 audio.

## 12. Dataset Details
**Not Applicable (Pretrained Model Inference).**  
This project is an inference application leveraging pre-trained foundation models. No custom dataset was collected for supervised training, and therefore no train/validation/test splits were generated. Standard sample audio files (e.g., `jfk.wav`) were utilized solely for functional verification of the pipeline.

## 13. API Documentation

### GET `/api/v1/health`
- **Description:** Verifies that the FastAPI backend is running.
- **Response:** `200 OK`
  ```json
  {"status": "healthy", "app_name": "AI Voice Translation Assistant", "environment": "development", "version": "1.0.0"}
  ```

### POST `/api/v1/translate-text`
- **Description:** Translates provided text and generates a TTS audio response.
- **Request Body (JSON):**
  ```json
  {"text": "Hello, how are you?", "source_language": "en", "target_language": "ur"}
  ```
- **Response:** `200 OK`
  ```json
  {
    "success": true,
    "translation": "ہیلو، آپ کیسے ہیں؟",
    "source_language": "en",
    "target_language": "ur",
    "audio_file": "generated_audio/tts_ur_12345.mp3",
    "processing_time": 21.5
  }
  ```

### POST `/api/v1/translate-audio`
- **Description:** Full STT → Translation → TTS pipeline for audio files.
- **Request (multipart/form-data):**
  - `audio_file`: The uploaded audio file.
  - `source_language`: Source language code (`en` or `ur`).
  - `target_language`: Target language code (`en` or `ur`).
- **Response:** `200 OK` (JSON payload including `transcription`, `translation`, and `audio_file`).

## 14. Input/Output Examples
- **Text Input Example:** 
  - *Input:* "Hello, how are you?" (English)
  - *Output:* "ہیلو، آپ کیسے ہیں؟" (Urdu) + Generated Urdu audio.
- **Audio Input Example:**
  - *Input:* User speaks into the microphone (English).
  - *Output:* The system transcribes the speech to text, translates it to Urdu, and plays back the translated Urdu audio.

## 15. Challenges Faced and Solutions
- **STT/Translation Parameter Mismatches:** Initial integration failed due to incorrect language code arguments being passed to `STTService.transcribe()` and `TranslationService.translate()`. This was solved by implementing a centralized `utils/language.py` module to normalize ISO codes.
- **Missing Piper TTS Assets:** The system crashed because Piper binaries and models were not included in the environment. Solved by implementing a robust `gTTS` fallback mechanism in `TTSService`.
- **Transformers Tokenizer Compatibility:** `transformers v5.14.1` deprecated the `lang_code_to_id` property, causing fatal `[Errno 22]` backend errors during translation. Resolved by refactoring the code to use `tokenizer.convert_tokens_to_ids()` natively.
- **Frontend Audio Format Mismatch:** The frontend was hardcoded to play `audio/wav`, but the `gTTS` fallback generates `.mp3`. Solved by dynamically updating the MIME type in Streamlit based on the generated file extension.
- **Backend Connection Issues:** Encountered port binding (`localhost` vs `127.0.0.1`) issues causing frontend disconnects (see testing report screenshots). Standardized API URLs in configuration.
- **Microphone Recording State Issues:** Browser security policies caused empty (0-byte) recordings. Solved by adding explicit 400 Bad Request error handling for empty files to prevent pipeline crashes.

## 16. Future Scope
- **GPU Acceleration:** Add CUDA support to drastically reduce the current ~28-second pipeline latency to real-time speeds.
- **Streaming Audio:** Implement WebSockets for real-time, chunked speech translation instead of batch processing whole files.
- **Broader Language Support:** Expand the language map to utilize more of the 200 languages supported by the NLLB model.

## 17. Contributors

- Hafiza Amna Naseem — Developer / Project Author

## 18. GitHub Repository Link

[AI Voice Translation Assistant - GitHub Repository](https://github.com/Hafiza-Amna/AI-Voice-Translation-Assistant)
