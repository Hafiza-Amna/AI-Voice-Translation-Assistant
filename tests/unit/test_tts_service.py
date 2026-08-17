"""Unit tests for TTSService using mocked Piper subprocess."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.tts_service import (
    TTSService,
    TTSServiceError,
    UnsupportedLanguageError,
)
from app.models.schemas import TTSResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a temporary output directory for generated audio files."""
    output_dir = tmp_path / "generated_audio"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Provide a temporary models directory with dummy voice model files."""
    models_dir = tmp_path / "piper_models"
    models_dir.mkdir()
    # Create dummy .onnx files so get_voice_model_path() succeeds
    (models_dir / "en_US-lessac-medium.onnx").write_text("dummy")
    (models_dir / "ur_PK-usman-medium.onnx").write_text("dummy")
    return models_dir


@pytest.fixture
def tts_service(tmp_output_dir, tmp_models_dir):
    """Create a TTSService with temp dirs and fixed voice config."""
    return TTSService(
        piper_executable="piper",
        models_dir=str(tmp_models_dir),
        english_voice="en_US-lessac-medium.onnx",
        urdu_voice="ur_PK-usman-medium.onnx",
        output_dir=str(tmp_output_dir),
    )


def make_piper_success():
    """Return a mock CompletedProcess simulating successful Piper execution."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = ""
    return mock_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("app.services.tts_service.subprocess.run")
def test_english_tts_synthesis(mock_run, tts_service):
    """Test successful English text-to-speech synthesis."""
    mock_run.return_value = make_piper_success()

    result = tts_service.synthesize("Hello, how are you?", language="en")

    assert isinstance(result, TTSResult)
    assert result.language == "en"
    assert result.audio_path.endswith(".wav")
    assert "tts_en_" in result.audio_path
    assert result.processing_time_seconds >= 0.0
    mock_run.assert_called_once()


@patch("app.services.tts_service.subprocess.run")
def test_urdu_tts_synthesis(mock_run, tts_service):
    """Test successful Urdu text-to-speech synthesis."""
    mock_run.return_value = make_piper_success()

    result = tts_service.synthesize("السلام علیکم", language="ur")

    assert isinstance(result, TTSResult)
    assert result.language == "ur"
    assert result.audio_path.endswith(".wav")
    assert "tts_ur_" in result.audio_path
    assert result.processing_time_seconds >= 0.0
    mock_run.assert_called_once()


@patch("app.services.tts_service.subprocess.run")
def test_empty_text_returns_empty_result(mock_run, tts_service):
    """Test that empty text returns TTSResult with empty audio_path and zero time."""
    result = tts_service.synthesize("   ", language="en")

    assert result.audio_path == ""
    assert result.language == "en"
    assert result.processing_time_seconds == 0.0
    # Piper subprocess must NOT be called for empty text
    mock_run.assert_not_called()


def test_unsupported_language_raises_error(tts_service):
    """Test that unsupported language raises UnsupportedLanguageError."""
    with pytest.raises(UnsupportedLanguageError):
        tts_service.synthesize("Bonjour", language="fr")


@patch("app.services.tts_service.subprocess.run")
def test_unique_audio_filenames(mock_run, tts_service):
    """Test that each synthesize call produces a unique output filename."""
    mock_run.return_value = make_piper_success()

    result1 = tts_service.synthesize("First sentence", language="en")
    result2 = tts_service.synthesize("Second sentence", language="en")

    assert result1.audio_path != result2.audio_path


@patch("app.services.tts_service.subprocess.run")
def test_processing_time_is_recorded(mock_run, tts_service):
    """Test that processing time is a non-negative float."""
    mock_run.return_value = make_piper_success()

    result = tts_service.synthesize("Test timing", language="en")

    assert isinstance(result.processing_time_seconds, float)
    assert result.processing_time_seconds >= 0.0


def test_configuration_loading():
    """Test that TTSService reads defaults from settings when no args given."""
    from app.config import settings
    service = TTSService()

    assert service.piper_executable == settings.PIPER_EXECUTABLE
    assert str(service.models_dir) == settings.PIPER_MODELS_DIR or \
           Path(settings.PIPER_MODELS_DIR).resolve() == service.models_dir.resolve() or \
           service.models_dir == Path(settings.PIPER_MODELS_DIR)
    assert service.english_voice == settings.PIPER_ENGLISH_VOICE
    assert service.urdu_voice == settings.PIPER_URDU_VOICE


@patch("app.services.tts_service.subprocess.run")
def test_piper_failure_raises_tts_service_error(mock_run, tts_service):
    """Test that a non-zero Piper exit code raises TTSServiceError."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "model not found"
    mock_run.return_value = mock_result

    with pytest.raises(TTSServiceError) as exc_info:
        tts_service.synthesize("Hello", language="en")

    assert "Piper synthesis failed" in str(exc_info.value)


@patch("app.services.tts_service.subprocess.run", side_effect=FileNotFoundError)
def test_piper_not_installed_raises_tts_service_error(mock_run, tts_service):
    """Test that a missing Piper binary raises TTSServiceError with clear message."""
    with pytest.raises(TTSServiceError) as exc_info:
        tts_service.synthesize("Hello", language="en")

    assert "Piper executable not found" in str(exc_info.value)


@patch("app.services.tts_service.subprocess.run")
def test_language_alias_accepted(mock_run, tts_service):
    """Test that full language names like 'english' and 'urdu' are accepted."""
    mock_run.return_value = make_piper_success()

    result_en = tts_service.synthesize("Hello", language="english")
    result_ur = tts_service.synthesize("السلام", language="urdu")

    assert result_en.language == "en"
    assert result_ur.language == "ur"
