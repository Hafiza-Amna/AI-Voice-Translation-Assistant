"""Audio validation and utility functions."""

from pathlib import Path
from typing import Union

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".amr", ".aac", ".flac", ".webm", ".3gp", ".mp4"}

class AudioValidationError(Exception):
    """Exception raised when audio file validation fails."""
    pass

def validate_audio_file(file_path: Union[str, Path]) -> Path:
    """
    Validates that an audio file exists, is non-empty, and has a supported format.

    Args:
        file_path: Path to the audio file.

    Returns:
        Path: Resolved Path object of the valid audio file.

    Raises:
        AudioValidationError: If the file does not exist, is empty, or is an unsupported format.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise AudioValidationError(f"Audio file not found: '{file_path}'")
    
    if not path.is_file():
        raise AudioValidationError(f"Path is not a file: '{file_path}'")
        
    if path.stat().st_size == 0:
        raise AudioValidationError(f"Audio file is empty: '{file_path}'")
        
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported audio format '{path.suffix}'. Supported formats are: {supported_str}"
        )
        
    return path
