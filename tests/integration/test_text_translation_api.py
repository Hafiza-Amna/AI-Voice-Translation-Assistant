"""Integration tests for POST /api/v1/translate-text endpoint."""

# pyrefly: ignore [missing-import]
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.api.dependencies import (
    get_translation_service,
    get_tts_service,
)
from app.main import app
from app.models.schemas import TranslationResult, TTSResult
from app.services.translation_service import TranslationServiceError
from app.services.tts_service import TTSServiceError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    dummy_wav = tmp_path / "tts_en_abcdef12.wav"
    dummy_wav.write_bytes(b"RIFF dummy wav content")
    service.synthesize.return_value = TTSResult(
        audio_path=str(dummy_wav),
        language="en",
        processing_time_seconds=0.2,
    )
    return service


@pytest.fixture(autouse=True)
def override_dependencies(mock_translation_service, mock_tts_service):
    """Override service singletons with mocks for these integration tests."""
    app.dependency_overrides[get_translation_service] = lambda: mock_translation_service
    app.dependency_overrides[get_tts_service] = lambda: mock_tts_service
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_translate_text_success(async_client):
    """POST /translate-text returns 200 with translation and audio_file."""
    payload = {
        "text": "السلام علیکم",
        "source_language": "ur",
        "target_language": "en",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["translation"] == "Peace be upon you"
    assert data["source_language"] == "ur"
    assert data["target_language"] == "en"
    assert "tts_en_" in data["audio_file"]
    assert isinstance(data["processing_time"], float)


@pytest.mark.asyncio
async def test_translate_text_english_to_urdu(async_client, mock_translation_service, mock_tts_service, tmp_path):
    """POST /translate-text works for en → ur direction."""
    mock_translation_service.translate.return_value = TranslationResult(
        original_text="Hello world",
        translated_text="ہیلو دنیا",
        source_language="en",
        target_language="ur",
        processing_time_seconds=0.25,
    )
    dummy_wav = tmp_path / "tts_ur_12345678.wav"
    dummy_wav.write_bytes(b"RIFF dummy ur wav")
    mock_tts_service.synthesize.return_value = TTSResult(
        audio_path=str(dummy_wav),
        language="ur",
        processing_time_seconds=0.15,
    )

    payload = {"text": "Hello world", "source_language": "en", "target_language": "ur"}
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["translation"] == "ہیلو دنیا"
    assert data["source_language"] == "en"
    assert data["target_language"] == "ur"


@pytest.mark.asyncio
async def test_translate_text_invalid_source_language(async_client):
    """POST /translate-text returns 422 for unsupported source language."""
    payload = {
        "text": "Bonjour",
        "source_language": "fr",
        "target_language": "en",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 422
    assert "unsupported source language" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_text_invalid_target_language(async_client):
    """POST /translate-text returns 422 for unsupported target language."""
    payload = {
        "text": "Hello",
        "source_language": "en",
        "target_language": "es",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 422
    assert "unsupported target language" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_text_translation_failure(async_client, mock_translation_service):
    """POST /translate-text returns 500 when TranslationService raises."""
    mock_translation_service.translate.side_effect = TranslationServiceError("NLLB model error")

    payload = {
        "text": "السلام علیکم",
        "source_language": "ur",
        "target_language": "en",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 500
    assert "text translation failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_text_tts_failure(async_client, mock_tts_service):
    """POST /translate-text returns 500 when TTSService raises."""
    mock_tts_service.synthesize.side_effect = TTSServiceError("Piper not found")

    payload = {
        "text": "السلام علیکم",
        "source_language": "ur",
        "target_language": "en",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 500
    assert "text-to-speech synthesis failed" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_text_missing_generated_audio(async_client, mock_tts_service):
    """POST /translate-text returns 404 when TTS audio file does not exist on disk."""
    mock_tts_service.synthesize.return_value = TTSResult(
        audio_path="/non/existent/path/tts_en_missing.wav",
        language="en",
        processing_time_seconds=0.1,
    )

    payload = {
        "text": "السلام علیکم",
        "source_language": "ur",
        "target_language": "en",
    }
    response = await async_client.post("/api/v1/translate-text", json=payload)

    assert response.status_code == 404
    assert "generated audio file not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translate_text_empty_text(async_client, mock_translation_service, mock_tts_service):
    """POST /translate-text handles empty text gracefully (translation service returns empty result)."""
    mock_translation_service.translate.return_value = TranslationResult(
        original_text="",
        translated_text="",
        source_language="ur",
        target_language="en",
        processing_time_seconds=0.0,
    )
    # TTS with empty text returns empty audio_path
    mock_tts_service.synthesize.return_value = TTSResult(
        audio_path="",
        language="en",
        processing_time_seconds=0.0,
    )

    payload = {"text": "", "source_language": "ur", "target_language": "en"}
    response = await async_client.post("/api/v1/translate-text", json=payload)

    # Should succeed (empty but valid)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["translation"] == ""
