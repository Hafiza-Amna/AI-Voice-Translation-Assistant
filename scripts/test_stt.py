"""
Manual testing utility script for Speech-to-Text (faster-whisper) service.

Usage:
    python scripts/test_stt.py <path_to_audio_file>
"""

import sys
import logging
from pathlib import Path

# Add project root directory to sys.path so app package can be resolved
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.stt_service import STTService, STTServiceError
from app.utils.audio import AudioValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_stt.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"\n--- STT Manual Test ---")
    print(f"Audio file: {audio_path}")

    try:
        service = STTService()
        print(f"Initializing STTService (model: '{service.model_size}', device: '{service.device}', compute: '{service.compute_type}')...")
        
        result = service.transcribe(audio_path)

        print("\n--- Transcription Result ---")
        print(f"Detected Language   : {result.language}")
        print(f"Confidence Score    : {result.language_probability:.4f}")
        print(f"Processing Time     : {result.processing_time_seconds:.4f} seconds")
        print(f"Transcribed Text    :\n{result.text}")
        print("----------------------------\n")

    except AudioValidationError as ave:
        print(f"\n[Validation Error] {ave}")
        sys.exit(1)
    except STTServiceError as se:
        print(f"\n[STT Service Error] {se}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
