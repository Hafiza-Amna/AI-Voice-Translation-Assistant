"""Speech-to-Text service wrapper using faster-whisper."""

import time
import logging
from pathlib import Path
from typing import Union, Optional
from faster_whisper import WhisperModel

from app.config import settings
from app.models.schemas import STTResult
from app.utils.audio import validate_audio_file, AudioValidationError
from app.utils.language import normalize_language_code

logger = logging.getLogger(__name__)


class STTServiceError(Exception):
    """Base exception for Speech-to-Text service errors."""
    pass


class STTService:
    """Speech-to-Text service wrapping faster-whisper."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        """
        Initialize STT service with specified parameters or settings defaults.
        
        Args:
            model_size: Whisper model size (e.g. 'small', 'base').
            device: Computing device ('cpu', 'cuda').
            compute_type: Quantization compute type ('int8', 'float16').
        """
        self.model_size = model_size or settings.WHISPER_MODEL_SIZE
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self._model: Optional[WhisperModel] = None

    def load_model(self) -> WhisperModel:
        """
        Loads and returns the WhisperModel if not already loaded.
        Ensures model is loaded only once per service instance.
        """
        if self._model is None:
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' "
                f"on device '{self.device}' with compute_type '{self.compute_type}'..."
            )
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                logger.info("faster-whisper model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load faster-whisper model: {e}")
                raise STTServiceError(f"Model initialization failed: {e}") from e
        return self._model

    def transcribe(self, audio_path: Union[str, Path]) -> STTResult:
        """
        Transcribes audio file and returns detected language, text, confidence, and timing.

        Args:
            audio_path: Path to the audio file.

        Returns:
            STTResult: Object containing text, language, language_probability, and processing_time_seconds.

        Raises:
            AudioValidationError: If the audio file is missing, empty, or an invalid format.
            STTServiceError: If model transcription fails.
        """
        validated_path = validate_audio_file(audio_path)
        model = self.load_model()

        start_time = time.perf_counter()
        try:
            segments, info = model.transcribe(str(validated_path))
            
            # segments is a generator, iterate to collect transcribed text
            text_segments = [segment.text for segment in segments]
            full_text = " ".join(text_segments).strip()
            
            elapsed_time = round(time.perf_counter() - start_time, 4)
            detected_lang = normalize_language_code(getattr(info, "language", ""))
            prob = float(getattr(info, "language_probability", 0.0))

            logger.info(
                f"Transcription completed in {elapsed_time}s. "
                f"Detected language: '{detected_lang}' (p={prob:.2f})."
            )

            return STTResult(
                text=full_text,
                language=detected_lang,
                language_probability=prob,
                processing_time_seconds=elapsed_time,
            )
        except AudioValidationError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed for '{validated_path}': {e}")
            raise STTServiceError(f"Transcription execution failed: {e}") from e
