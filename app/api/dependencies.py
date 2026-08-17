"""Dependency injection getters for service singletons."""

from functools import lru_cache
from app.services.stt_service import STTService
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService


@lru_cache()
def get_stt_service() -> STTService:
    """Provides a singleton instance of STTService."""
    return STTService()


@lru_cache()
def get_translation_service() -> TranslationService:
    """Provides a singleton instance of TranslationService."""
    return TranslationService()


@lru_cache()
def get_tts_service() -> TTSService:
    """Provides a singleton instance of TTSService."""
    return TTSService()
