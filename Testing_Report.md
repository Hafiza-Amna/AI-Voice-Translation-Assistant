# Testing Report: AI Voice Translation Assistant

## 1. Project Title and Objective
**Project Title:** AI Voice Translation Assistant (Urdu ↔ English)  
**Objective:** To provide an end-to-end, privacy-focused voice translation system that converts spoken Urdu to English and spoken English to Urdu using state-of-the-art pretrained machine learning models.

## 2. Description of Dataset(s) Used
**Not Applicable (Pretrained Model Inference).**  
This project does not involve custom supervised training of a classification or regression model. Therefore, traditional dataset metrics such as size, source, and train/validation/test splits are not applicable. 

Instead, the system acts as an inference pipeline leveraging foundation models:
- **Speech-to-Text:** `faster-whisper` (pretrained on multilingual audio).
- **Translation:** `NLLB-200-distilled-600M` (pretrained by Meta on 200 languages).
- **Text-to-Speech:** `gTTS` / `Piper` (pretrained speech synthesizers).

For functional verification, manual sample audio files (e.g., `jfk.wav`, English speech, 352KB) and manual text inputs were utilized.

## 3. Data Preprocessing Steps
While no model training preprocessing was done, inference preprocessing includes:
- **Audio Validation:** The backend validates file existence, size (max 10 MB), and extension type (`.wav`, `.mp3`, `.m4a`, `.ogg`).
- **Tokenization:** Text is tokenized using the NLLB tokenizer. `convert_tokens_to_ids()` is explicitly used to map language codes (`urd_Arab`, `eng_Latn`) to token IDs.
- **Audio Resampling:** Handled automatically by the Streamlit frontend and `faster-whisper` engine to match the required 16kHz sample rate for Whisper.

## 4. Testing Methodology
The testing strategy utilized a combination of automated unit/integration testing and manual end-to-end testing:
- **Automated Testing:** `pytest` was utilized with the `httpx.AsyncClient` to simulate FastAPI requests and mock backend AI services.
- **End-to-End Pipeline Verification:** A dedicated testing script (`e2e_verify.py`) executed the real pipeline (STT → Translate → TTS) using real audio files.
- **Manual UI Testing:** The Streamlit frontend was manually tested across different scenarios (recording, uploading, text input).

## 5. Multiple Test Cases and Scenarios

| Test ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| TC-01 | Backend Health Check | GET `/api/v1/health` | HTTP 200, `{"status": "healthy"}` | HTTP 200, `{"status": "healthy"}` | PASS |
| TC-02 | Valid Text Translation | POST `{"text": "آپ کیسے ہیں؟", "source_language": "ur", "target_language": "en"}` | HTTP 200, translated text `"How are you?"`, audio file path | HTTP 200, text `"How are you?"`, MP3 generated | PASS |
| TC-03 | Empty Text Translation | POST `{"text": ""...}` | HTTP 200, empty translation, empty audio | HTTP 200, empty fields returned | PASS |
| TC-04 | Audio Upload Validation (Empty File) | Upload 0-byte `.wav` | HTTP 400 Bad Request | HTTP 400, "Empty audio file" | PASS |
| TC-05 | Audio Upload Validation (Invalid Format) | Upload `.txt` file | HTTP 400 Bad Request | HTTP 400, "Unsupported audio format" | PASS |
| TC-06 | End-to-End Audio Pipeline | Upload `jfk.wav` (English to Urdu) | HTTP 200, English transcript, Urdu translation, Urdu MP3 | HTTP 200, valid transcript, translation, and MP3 | PASS |
| TC-07 | STT Transcribe Failure Handling | Mock STT failure | HTTP 500 Internal Server Error | HTTP 500 | PASS |

## 6. Testing on Different Datasets
**Not Applicable.** Functional testing was conducted on ad-hoc recordings and standard sample audio files rather than formal benchmark datasets.

## 7. Model Evaluation Metrics
**Not Applicable.** Because this project implements pretrained models rather than training a custom classifier, standard metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC are not generated. 

The primary evaluation metric for this software engineering project is the **Test Pass Rate**:
- **Current Pytest Result:** **54 / 54 tests passed (100%)**

## 8. Confusion Matrix / Evaluation Graphs
**Not Applicable.**

## 9. Comparison with Baseline or Previous Models
**Not Applicable.**

## 10. Error Analysis (Issues Found During Testing)

During the development and testing phase, several errors were encountered and resolved.

| Component | Error Encountered | Possible Reason / Root Cause | Solution Implemented |
|---|---|---|---|
| **STT Service** | `faster-whisper` crashed when transcribing | Passed incorrect language parameter format to `transcribe()`. | Removed hardcoded language to let Whisper auto-detect, or pass valid ISO codes. |
| **Translation** | `[Errno 22]` or `KeyError` during pipeline execution | `transformers v5.14.1` deprecated `lang_code_to_id`. | Refactored `TranslationService` to use `convert_tokens_to_ids()` natively. |
| **TTS Service** | Pipeline failure during voice generation | `piper` binary and models were missing from the host environment. | Implemented a robust fallback mechanism using Google TTS (`gTTS`). |
| **Frontend** | Translated audio would not play in Streamlit | `gTTS` generated `.mp3`, but frontend hardcoded `audio/wav` MIME type. | Added dynamic MIME type detection based on the file extension. |
| **Microphone** | Voice recording captured 0-byte files | Browser security/session state issues with the Streamlit audio component. | Confirmed HTTP/HTTPS permission requirements; added robust empty file validation (HTTP 400) in the backend. |
| **API Connection** | Backend Connection Issue (See Screenshot) | Localhost vs 127.0.0.1 mapping or server crash during model loading. | Standardized connection URLs and fixed backend fatal crashes (Tokenizer issue). |

## 11. Performance Statistics

*Note: These are observed metrics from local development runs on a standard CPU environment.*

| Metric | Measured Value | Notes |
|---|---|---|
| **Text Translation Time** | ~21 seconds | Includes NLLB-200 model loading and inference. |
| **Complete Audio Translation Time** | ~28 seconds | Includes STT, Translation, and TTS execution. |
| **STT Inference Time** | ~10.5 seconds | For a ~350KB WAV file. |
| **TTS Synthesis Time (gTTS)** | ~3.4 seconds | Highly dependent on network latency. |
| **Memory Usage (RAM)** | ~1.5 GB - 2 GB | Required to load Whisper (small) and NLLB-200 simultaneously into memory. |

## 12. Screenshots of Outputs/Results

### Error Analysis / Issues Found During Testing
![Backend Connection Issue](screenshots/backend_connection_issue.png)
*Figure 1: Example of a backend connection issue encountered during early testing phases before environment standardization.*

### Final Successful Results
![Voice Translation Result](screenshots/voice_translation_result.png)
*Figure 2: Successful end-to-end voice recording, transcription, translation, and audio generation.*

![Text Translation Result](screenshots/text_translation_result.png)
*Figure 3: Successful text translation utilizing the ChatGPT-style conversation interface.*

## 13. Conclusion and Future Improvements
**Conclusion:** 
The AI Voice Translation Assistant was successfully implemented and verified. The backend pipeline is highly reliable, cleanly passing all 54 automated integration and unit tests. Challenges related to library deprecations (NLLB tokenizer) and missing dependencies (Piper TTS) were successfully mitigated through robust error handling and fallback mechanisms (gTTS). The final application successfully provides end-to-end speech translation.

**Future Improvements:**
1. Hardware acceleration (GPU/CUDA support) to reduce the ~28 second latency to sub-second real-time performance.
2. Replacing the `gTTS` fallback with a fully packaged offline Piper TTS environment.
3. Implementing WebSocket streaming for real-time chunked translation instead of waiting for the full audio file to finish.
