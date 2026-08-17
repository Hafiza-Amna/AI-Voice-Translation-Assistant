"""Unit tests for STTService with mocked faster-whisper model."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.stt_service import STTService, STTServiceError
from app.utils.audio import AudioValidationError
from app.models.schemas import STTResult


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Fixture providing a temporary valid audio file."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(b"dummy audio header and PCM payload")
    return audio_path


class MockSegment:
    def __init__(self, text: str):
        self.text = text


class MockInfo:
    def __init__(self, language: str, language_probability: float):
        self.language = language
        self.language_probability = language_probability


def test_stt_service_initialization():
    """Test initializing STTService sets default configuration options."""
    service = STTService(model_size="small", device="cpu", compute_type="int8")
    assert service.model_size == "small"
    assert service.device == "cpu"
    assert service.compute_type == "int8"
    assert service._model is None


@patch("app.services.stt_service.WhisperModel")
def test_transcribe_urdu_success(mock_whisper_cls, sample_wav: Path):
    """Test successful Urdu speech transcription and language detection."""
    mock_model_instance = MagicMock()
    mock_whisper_cls.return_value = mock_model_instance

    mock_segments = [MockSegment("السلام علیکم"), MockSegment("آپ کیسے ہیں؟")]
    mock_info = MockInfo(language="ur", language_probability=0.98)
    mock_model_instance.transcribe.return_value = (mock_segments, mock_info)

    service = STTService()
    result = service.transcribe(sample_wav)

    assert isinstance(result, STTResult)
    assert result.text == "السلام علیکم آپ کیسے ہیں؟"
    assert result.language == "ur"
    assert result.language_probability == 0.98
    assert result.processing_time_seconds >= 0.0
    mock_model_instance.transcribe.assert_called_once_with(str(sample_wav))


@patch("app.services.stt_service.WhisperModel")
def test_transcribe_english_success(mock_whisper_cls, sample_wav: Path):
    """Test successful English speech transcription and language detection."""
    mock_model_instance = MagicMock()
    mock_whisper_cls.return_value = mock_model_instance

    mock_segments = [MockSegment("Hello"), MockSegment("how are you?")]
    mock_info = MockInfo(language="en", language_probability=0.99)
    mock_model_instance.transcribe.return_value = (mock_segments, mock_info)

    service = STTService()
    result = service.transcribe(sample_wav)

    assert isinstance(result, STTResult)
    assert result.text == "Hello how are you?"
    assert result.language == "en"
    assert result.language_probability == 0.99


@patch("app.services.stt_service.WhisperModel")
def test_transcribe_empty_text_handling(mock_whisper_cls, sample_wav: Path):
    """Test handling of audio containing no audible speech."""
    mock_model_instance = MagicMock()
    mock_whisper_cls.return_value = mock_model_instance

    mock_segments = []
    mock_info = MockInfo(language="en", language_probability=0.50)
    mock_model_instance.transcribe.return_value = (mock_segments, mock_info)

    service = STTService()
    result = service.transcribe(sample_wav)

    assert result.text == ""
    assert result.language == "en"


def test_transcribe_missing_file():
    """Test exception raised for missing audio file without loading model."""
    service = STTService()
    with pytest.raises(AudioValidationError, match="not found"):
        service.transcribe("non_existent_file.wav")


def test_transcribe_invalid_format(tmp_path: Path):
    """Test exception raised for unsupported file format."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_bytes(b"hello text")

    service = STTService()
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        service.transcribe(invalid_file)


@patch("app.services.stt_service.WhisperModel")
def test_model_loaded_only_once(mock_whisper_cls, sample_wav: Path):
    """Test model is instantiated only once across multiple transcription requests."""
    mock_model_instance = MagicMock()
    mock_whisper_cls.return_value = mock_model_instance
    mock_model_instance.transcribe.return_value = ([], MockInfo("en", 0.9))

    service = STTService()
    service.transcribe(sample_wav)
    service.transcribe(sample_wav)

    assert mock_whisper_cls.call_count == 1


@patch("app.services.stt_service.WhisperModel")
def test_model_init_failure(mock_whisper_cls, sample_wav: Path):
    """Test STTServiceError is raised if model initialization fails."""
    mock_whisper_cls.side_effect = Exception("Model load error")

    service = STTService()
    with pytest.raises(STTServiceError, match="Model initialization failed"):
        service.transcribe(sample_wav)
