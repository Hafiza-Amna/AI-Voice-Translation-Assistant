import json
import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def main():
    # Patch tts to bypass piper error
    from app.services.tts_service import TTSService
    from app.models.schemas import TTSResult
    def mock_synthesize(self, text, language):
        with open("mock.wav", "w") as f:
            f.write("mock")
        return TTSResult(audio_path="mock.wav", language=language, processing_time_seconds=0.1)
    TTSService.synthesize = mock_synthesize

    with open("jfk.wav", "rb") as f:
        audio_bytes = f.read()

    response = client.post(
        "/api/v1/translate-audio",
        files={"audio_file": ("jfk.wav", audio_bytes, "audio/wav")},
        data={"source_language": "en", "target_language": "ur"},
    )
    
    with open("client_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": response.status_code,
            "json": response.json()
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
