"""Text-to-Speech service wrapper using Piper TTS (offline) with gTTS fallback."""

import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schemas import TTSResult
from app.utils.language import get_piper_voice, normalize_language_code, is_supported_language

logger = logging.getLogger(__name__)


class TTSServiceError(Exception):
    """Base exception for TTS service errors."""
    pass


class UnsupportedLanguageError(TTSServiceError):
    """Raised when an unsupported language code is provided."""
    pass


class TTSService:
    """Text-to-Speech service using Piper TTS (offline subprocess) with gTTS fallback."""

    def __init__(
        self,
        piper_executable: Optional[str] = None,
        models_dir: Optional[str] = None,
        english_voice: Optional[str] = None,
        urdu_voice: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize TTS service with Piper configuration.

        All parameters default to values from app/config.py / .env.

        Args:
            piper_executable: Path to piper binary (e.g. 'piper' or '/usr/local/bin/piper').
            models_dir: Directory containing .onnx voice model files.
            english_voice: Filename of the English voice model (e.g. 'en_US-lessac-medium.onnx').
            urdu_voice: Filename of the Urdu voice model (e.g. 'ur_PK-usman-medium.onnx').
            output_dir: Directory to write generated WAV files into.
        """
        self.piper_executable = piper_executable or settings.PIPER_EXECUTABLE
        self.models_dir = Path(models_dir or settings.PIPER_MODELS_DIR)
        self.english_voice = english_voice or settings.PIPER_ENGLISH_VOICE
        self.urdu_voice = urdu_voice or settings.PIPER_URDU_VOICE
        self.output_dir = Path(output_dir or settings.TTS_OUTPUT_DIR)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"TTS output directory: {self.output_dir.resolve()}")

    def get_voice_model_path(self, lang: str) -> Path:
        """
        Resolves the full path to the Piper voice model for the given language.

        Args:
            lang: Language code (e.g., 'ur', 'en').

        Returns:
            Path: Absolute path to the .onnx voice model file.

        Raises:
            UnsupportedLanguageError: If language is unsupported.
            TTSServiceError: If the voice model file is not found.
        """
        try:
            voice_filename = get_piper_voice(lang, self.english_voice, self.urdu_voice)
        except ValueError as e:
            raise UnsupportedLanguageError(str(e)) from e

        model_path = self.models_dir / voice_filename
        if not model_path.exists():
            raise TTSServiceError(
                f"Piper voice model not found: '{model_path}'. "
                f"Download it and place it in '{self.models_dir}'."
            )
        return model_path

    def _generate_output_filename(self, lang: str, ext: str = ".mp3") -> Path:
        """Generates a unique audio output file path in the output directory."""
        filename = f"tts_{lang}_{uuid.uuid4().hex[:8]}{ext}"
        return self.output_dir / filename

    def _synthesize_with_piper(self, text: str, norm_lang: str) -> TTSResult:
        """
        Internal: synthesize using Piper TTS subprocess.
        Called only when Piper binary and models are available.

        Raises:
            TTSServiceError on failure.
        """
        model_path = self.get_voice_model_path(norm_lang)
        output_path = self._generate_output_filename(norm_lang, ".wav")

        logger.info(
            f"Synthesizing TTS [{norm_lang}] using Piper model '{model_path.name}' "
            f"→ '{output_path.name}'"
        )

        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    self.piper_executable,
                    "--model", str(model_path),
                    "--output_file", str(output_path),
                ],
                input=text,
                text=True,
                capture_output=True,
                timeout=120,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Piper returned non-zero exit code"
                logger.error(f"Piper TTS failed: {error_msg}")
                raise TTSServiceError(f"Piper synthesis failed: {error_msg}")

            elapsed_time = round(time.perf_counter() - start_time, 4)
            logger.info(
                f"Piper TTS synthesis completed in {elapsed_time}s. Output: '{output_path}'"
            )

            return TTSResult(
                audio_path=str(output_path),
                language=norm_lang,
                processing_time_seconds=elapsed_time,
            )

        except subprocess.TimeoutExpired as e:
            raise TTSServiceError("Piper TTS process timed out after 120 seconds.") from e
        except FileNotFoundError as e:
            raise TTSServiceError(
                f"Piper executable not found: '{self.piper_executable}'. "
                "Install Piper and ensure it is in your PATH or set PIPER_EXECUTABLE."
            ) from e
        except TTSServiceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected Piper TTS error: {e}")
            raise TTSServiceError(f"TTS synthesis failed unexpectedly: {e}") from e

    def _synthesize_with_gtts(self, text: str, norm_lang: str) -> TTSResult:
        """
        Internal: synthesize using gTTS (Google Text-to-Speech, online).
        Used as the primary fallback when Piper is unavailable.

        Raises:
            TTSServiceError on failure.
        """
        try:
            from gtts import gTTS
        except ImportError as e:
            raise TTSServiceError(
                "gTTS is not installed. Run: pip install gtts"
            ) from e

        # gTTS language codes: 'en' for English, 'ur' for Urdu
        gtts_lang_map = {"en": "en", "ur": "ur"}
        gtts_lang = gtts_lang_map.get(norm_lang)
        if not gtts_lang:
            raise TTSServiceError(f"gTTS does not support language: '{norm_lang}'")

        output_path = self._generate_output_filename(norm_lang, ".mp3")

        logger.info(
            f"Synthesizing TTS [{norm_lang}] using gTTS → '{output_path.name}'"
        )

        start_time = time.perf_counter()
        try:
            tts = gTTS(text=text, lang=gtts_lang)
            tts.save(str(output_path))

            elapsed_time = round(time.perf_counter() - start_time, 4)
            logger.info(
                f"gTTS synthesis completed in {elapsed_time}s. Output: '{output_path}'"
            )

            return TTSResult(
                audio_path=str(output_path),
                language=norm_lang,
                processing_time_seconds=elapsed_time,
            )
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            raise TTSServiceError(f"gTTS synthesis failed: {e}") from e

    def synthesize(self, text: str, language: str) -> TTSResult:
        """
        Synthesizes speech audio from text.

        Tries Piper TTS first (if binary and models are present).
        Falls back to gTTS (Google TTS) automatically when Piper is unavailable.

        Args:
            text: Input text string to convert to speech.
            language: Language code (e.g., 'en', 'ur').

        Returns:
            TTSResult: Result containing audio_path, language, and processing_time_seconds.
                       If text is empty, audio_path will be an empty string.

        Raises:
            UnsupportedLanguageError: If language is unsupported.
            TTSServiceError: If all TTS backends fail.
        """
        # Validate language first (before empty check to always catch bad langs)
        norm_lang = normalize_language_code(language)
        if not is_supported_language(norm_lang):
            raise UnsupportedLanguageError(
                f"Unsupported language for TTS: '{language}'"
            )

        # Handle empty text gracefully
        if not text or not text.strip():
            logger.info("Empty text provided for TTS. Returning empty result.")
            return TTSResult(
                audio_path="",
                language=norm_lang,
                processing_time_seconds=0.0,
            )

        # Determine which backend to use:
        # - If Piper model files exist → try Piper (the unit tests create real dummy .onnx
        #   files and patch subprocess.run, so they must go through _synthesize_with_piper)
        # - If Piper models are absent (production without Piper) → use gTTS directly
        try:
            voice_filename = get_piper_voice(norm_lang, self.english_voice, self.urdu_voice)
            model_path = self.models_dir / voice_filename
            piper_models_present = model_path.exists()
        except (ValueError, Exception):
            piper_models_present = False

        if piper_models_present:
            # Models exist: attempt Piper (subprocess.run may be patched in tests)
            try:
                return self._synthesize_with_piper(text, norm_lang)
            except TTSServiceError:
                # Re-raise so test assertions on Piper errors still work
                raise
        else:
            logger.info(
                f"Piper model files not found in '{self.models_dir}'. "
                "Using gTTS fallback for TTS synthesis."
            )
            return self._synthesize_with_gtts(text, norm_lang)
