"""Unit tests for audio validation utility."""

import pytest
from pathlib import Path
from app.utils.audio import validate_audio_file, AudioValidationError

def test_validate_audio_file_success(tmp_path: Path):
    """Test successful validation of a valid audio file."""
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"dummy audio content")

    result = validate_audio_file(audio_file)
    assert result == audio_file

def test_validate_audio_file_not_found(tmp_path: Path):
    """Test validation failure when file does not exist."""
    non_existent = tmp_path / "non_existent.wav"
    with pytest.raises(AudioValidationError, match="not found"):
        validate_audio_file(non_existent)

def test_validate_audio_file_empty(tmp_path: Path):
    """Test validation failure when file is empty."""
    empty_file = tmp_path / "empty.mp3"
    empty_file.touch()

    with pytest.raises(AudioValidationError, match="is empty"):
        validate_audio_file(empty_file)

def test_validate_audio_file_unsupported_format(tmp_path: Path):
    """Test validation failure for unsupported extension."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_bytes(b"text content")

    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        validate_audio_file(invalid_file)

@pytest.mark.parametrize("ext", [".wav", ".mp3", ".m4a", ".ogg"])
def test_validate_supported_formats(tmp_path: Path, ext: str):
    """Test all supported formats are accepted."""
    file_path = tmp_path / f"test{ext}"
    file_path.write_bytes(b"data")
    assert validate_audio_file(file_path) == file_path
