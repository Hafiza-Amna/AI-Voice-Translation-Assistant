#!/usr/bin/env python3
"""Manual TTS script for testing Piper TTS service."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.tts_service import (
    TTSService,
    TTSServiceError,
    UnsupportedLanguageError,
)


def main():
    parser = argparse.ArgumentParser(description="Test Piper TTS Service")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text string to synthesize into speech",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code (e.g. 'en', 'ur', 'english', 'urdu')",
    )

    args = parser.parse_args()

    try:
        service = TTSService()
        result = service.synthesize(text=args.text, language=args.language)

        if not result.audio_path:
            print("No audio generated (empty text provided).")
            return

        print("\n--- TTS Result ---")
        print(f"Language:        {result.language}")
        print(f"Output Path:     {result.audio_path}")
        print(f"Processing Time: {result.processing_time_seconds:.4f}s")
        print("------------------\n")

    except UnsupportedLanguageError as e:
        print(f"Error: Unsupported language — {e}", file=sys.stderr)
        sys.exit(1)
    except TTSServiceError as e:
        print(f"Error: TTS failed — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
