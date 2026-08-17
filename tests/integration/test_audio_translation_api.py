"""Integration tests for POST /translate-audio endpoint."""

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.api.dependencies import (
    get_stt_service,
    get_translation_service,
    get_tts_service,
)
from app.main import app
from app.models.schemas import STTResult, TranslationResult, TTSResult
from app.services.stt_service import STTServiceError
from app.services.translation_service import TranslationServiceError
from app.services.tts_service import TTSServiceError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stt_service():
    service = MagicMock()
    service.transcribe.return_value = STTResult(
        text="السلام علیکم",
        language="ur",
        language_probability=0.99,
        processing_time_seconds=0.5,
    )
    return service


@pytest.fixture
def mock_translation_service():
    service = MagicMock()
    service.translate.return_value = TranslationResult(
        original_text="السلام علیکم",
        translated_text="Peace be upon you",
        source_language="ur",
        target_language="en",
        processing_time_seconds=0.3,
    )
    return service


@pytest.fixture
def mock_tts_service(tmp_path):
    service = MagicMock()
    # Create a dummy generated audio file so path.exists() passes
    dummy_wav = tmp_path / "tts_en_12345678.wav"
    dummy_wav.write_bytes(b"RIFF dummy wav content")

    service.synthesize.return_value = TTSResult(
        audio_path=str(dummy_wav),
        language="en",
        processing_time_seconds=0.4,
    )
    return service


@pytest.fixture(autouse=True)
def override_dependencies(mock_stt_service, mock_translation_service, mock_tts_service):
    """Override service singletons with mocks for integration tests."""
    app.dependency_overrides[get_stt_service] = lambda: mock_stt_service
    app.dependency_overrides[get_translation_service] = lambda: mock_translation_service
    app.dependency_overrides[get_tts_service] = lambda: mock_tts_service
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_translate_audio_success(async_client):
    """Test successful end-to-end audio translation (200 OK)."""
    audio_content = b"RIFF dummy wav audio content"
    files = {"audio_file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["success"] is True
    assert json_resp["transcription"] == "السلام علیکم"
    assert json_resp["translation"] == "Peace be upon you"
    assert json_resp["source_language"] == "ur"
    assert json_resp["target_language"] == "en"
    assert "tts_en_" in json_resp["audio_file"]
    assert isinstance(json_resp["processing_time"], float)


@pytest.mark.asyncio
async def test_translate_audio_empty_file(async_client):
    """Test empty audio upload returns 400 Bad Request."""
    files = {"audio_file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_unsupported_format(async_client):
    """Test unsupported file format returns 400 Bad Request."""
    files = {"audio_file": ("test.txt", io.BytesIO(b"some text content"), "text/plain")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 400
    assert "unsupported audio format" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_invalid_source_language(async_client):
    """Test invalid source language returns 422 Unprocessable Entity."""
    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "fr", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 422
    assert "unsupported source language" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_invalid_target_language(async_client):
    """Test invalid target language returns 422 Unprocessable Entity."""
    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "ur", "target_language": "es"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 422
    assert "unsupported target language" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_missing_generated_audio(async_client, mock_tts_service):
    """Test 404 error returned when generated audio file does not exist on disk."""
    mock_tts_service.synthesize.return_value = TTSResult(
        audio_path="/non/existent/path/generated.wav",
        language="en",
        processing_time_seconds=0.4,
    )

    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 404
    assert "generated audio file not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_stt_failure(async_client, mock_stt_service):
    """Test 500 status returned when STT service fails."""
    mock_stt_service.transcribe.side_effect = STTServiceError("Whisper failed to process")

    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 500
    assert "speech-to-text transcription failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_translation_failure(async_client, mock_translation_service):
    """Test 500 status returned when Translation service fails."""
    mock_translation_service.translate.side_effect = TranslationServiceError("NLLB model error")

    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 500
    assert "text translation failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_audio_tts_failure(async_client, mock_tts_service):
    """Test 500 status returned when TTS service fails."""
    mock_tts_service.synthesize.side_effect = TTSServiceError("Piper execution error")

    files = {"audio_file": ("test.wav", io.BytesIO(b"RIFF dummy content"), "audio/wav")}
    data = {"source_language": "ur", "target_language": "en"}

    response = await async_client.post("/api/v1/translate-audio", files=files, data=data)

    assert response.status_code == 500
    assert "text-to-speech synthesis failed" in response.json()["detail"].lower()
