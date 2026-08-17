import sys
from pathlib import Path
from app.services.stt_service import STTService
from app.services.translation_service import TranslationService

def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "jfk.wav"
    
    # 1. Inspect Audio
    p = Path(audio_path)
    if not p.exists():
        print(f"File not found: {audio_path}")
        return
        
    out = []
    out.append(f"File: {audio_path}")
    out.append(f"Size: {p.stat().st_size} bytes")

    # 2. STT
    out.append("\n--- STT Transcription ---")
    stt = STTService()
    try:
        res_stt = stt.transcribe(audio_path=audio_path)
        out.append(f"STT TEXT: '{res_stt.text}'")
        out.append(f"STT LANGUAGE: {res_stt.language}")
    except Exception as e:
        out.append(f"STT Error: {e}")
        with open("test_output.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        return

    # 3. Translation
    out.append("\n--- Text Translation ---")
    ts = TranslationService()
    try:
        res_trans = ts.translate(text=res_stt.text, source_lang="en", target_lang="ur")
        out.append(f"TRANSLATION INPUT: '{res_trans.original_text}'")
        out.append(f"TRANSLATION OUTPUT: '{res_trans.translated_text}'")
    except Exception as e:
        out.append(f"Translation Error: {e}")
        
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))

if __name__ == "__main__":
    main()
