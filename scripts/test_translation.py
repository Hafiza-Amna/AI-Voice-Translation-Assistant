#!/usr/bin/env python3
"""Manual translation script for testing NLLB-200 translation service."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.translation_service import (
    TranslationService,
    UnsupportedLanguageError,
    TranslationServiceError,
)


def main():
    parser = argparse.ArgumentParser(description="Test NLLB-200 Translation Service")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text string to translate",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source language code (e.g. 'ur', 'en')",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target language code (e.g. 'en', 'ur')",
    )

    args = parser.parse_args()

    try:
        service = TranslationService()
        result = service.translate(
            text=args.text,
            source_lang=args.source,
            target_lang=args.target,
        )

        print("\n--- Translation Result ---")
        print(f"Original:        {result.original_text}")
        print(f"Translated:      {result.translated_text}")
        print(f"Source Language: {result.source_language}")
        print(f"Target Language: {result.target_language}")
        print(f"Processing Time: {result.processing_time_seconds:.4f}s")
        print("--------------------------\n")
    except UnsupportedLanguageError as e:
        print(f"Error: Unsupported language - {e}", file=sys.stderr)
        sys.exit(1)
    except TranslationServiceError as e:
        print(f"Error: Translation failed - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
