import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def main():
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
