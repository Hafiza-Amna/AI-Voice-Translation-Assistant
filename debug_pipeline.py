import sys
from pathlib import Path

from app.services.tts_service import TTSService
from app.services.stt_service import STTService
from app.services.translation_service import TranslationService
from app.models.schemas import AudioTranslationResponse

def main():
    print("--- 1. Generating Test Audio (English) ---")
    tts = TTSService()
    test_text = "Hello world, this is a test of the speech recognition pipeline. I hope it works perfectly."
    res_tts1 = tts.synthesize(text=test_text, language="en")
    print(f"Generated: {res_tts1.audio_path}")
    
    # Check file size
    size = Path(res_tts1.audio_path).stat().st_size
    print(f"File size: {size} bytes")

    print("\n--- 2. STT Transcription ---")
    stt = STTService()
    res_stt = stt.transcribe(audio_path=res_tts1.audio_path)
    print(f"STT TEXT: '{res_stt.text}'")
    print(f"STT LANGUAGE: {res_stt.language}")
    print(f"STT SEGMENTS: {len(res_stt.segments)}")
    
    print("\n--- 3. Text Translation ---")
    ts = TranslationService()
    res_trans = ts.translate(text=res_stt.text, source_lang="en", target_lang="ur")
    print(f"TRANSLATION INPUT: '{res_trans.original_text}'")
    print(f"TRANSLATION OUTPUT: '{res_trans.translated_text}'")
    
    print("\n--- 4. TTS (Target Language) ---")
    res_tts2 = tts.synthesize(text=res_trans.translated_text, language="ur")
    print(f"Final audio path: {res_tts2.audio_path}")
    
    print("\n--- 5. Verify AudioTranslationResponse Schema ---")
    resp = AudioTranslationResponse(
        success=True,
        transcription=res_stt.text,
        translation=res_trans.translated_text,
        source_language="en",
        target_language="ur",
        audio_file=res_tts2.audio_path,
        processing_time=1.23,
    )
    print("AudioTranslationResponse Dump:")
    print(resp.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
