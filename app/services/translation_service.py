"""Translation service wrapper using Meta's NLLB-200 model."""

import time
import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import settings
from app.models.schemas import TranslationResult
from app.utils.language import (
    normalize_language_code,
    get_nllb_code,
    is_supported_language,
)

logger = logging.getLogger(__name__)


class TranslationServiceError(Exception):
    """Base exception for Translation service errors."""
    pass


class UnsupportedLanguageError(TranslationServiceError):
    """Raised when an unsupported language code is provided."""
    pass


class TranslationService:
    """Text translation service using Hugging Face NLLB-200."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize Translation service with model parameters or config defaults.

        Args:
            model_name: Hugging Face model identifier (e.g. 'facebook/nllb-200-distilled-600M').
            device: Computing device ('cpu', 'cuda').
        """
        self.model_name = model_name or settings.NLLB_MODEL_NAME
        self.device = device or settings.NLLB_DEVICE
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSeq2SeqLM] = None

    def load_model(self) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """
        Loads and returns the AutoTokenizer and AutoModelForSeq2SeqLM if not already loaded.
        Ensures model is loaded only once per service instance.
        """
        if self._tokenizer is None or self._model is None:
            logger.info(
                f"Loading NLLB-200 model '{self.model_name}' on device '{self.device}'..."
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                if hasattr(self._model, "to"):
                    self._model = self._model.to(self.device)
                logger.info("NLLB-200 model and tokenizer loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NLLB model '{self.model_name}': {e}")
                raise TranslationServiceError(f"Model initialization failed: {e}") from e

        return self._tokenizer, self._model

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> TranslationResult:
        """
        Translates text between supported languages (e.g. Urdu and English).

        Args:
            text: Input text string to translate.
            source_lang: Source language code (e.g., 'ur', 'en').
            target_lang: Target language code (e.g., 'en', 'ur').

        Returns:
            TranslationResult: Result containing original and translated text, languages, and timing.

        Raises:
            UnsupportedLanguageError: If source or target language is unsupported.
            TranslationServiceError: If translation execution fails.
        """
        # 1. Validate languages
        try:
            src_nllb = get_nllb_code(source_lang)
        except ValueError as e:
            raise UnsupportedLanguageError(f"Invalid source language: {e}") from e

        try:
            tgt_nllb = get_nllb_code(target_lang)
        except ValueError as e:
            raise UnsupportedLanguageError(f"Invalid target language: {e}") from e

        norm_src = normalize_language_code(source_lang)
        norm_tgt = normalize_language_code(target_lang)

        # 2. Handle empty/whitespace text gracefully
        if not text or not text.strip():
            logger.info("Empty text provided for translation. Returning empty result.")
            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language=norm_src,
                target_language=norm_tgt,
                processing_time_seconds=0.0,
            )

        start_time = time.perf_counter()
        try:
            tokenizer, model = self.load_model()

            # 3. Prepare tokenizer with source language
            tokenizer.src_lang = src_nllb
            inputs = tokenizer(text, return_tensors="pt")

            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)

            # 4. Get target language token ID and generate translation
            if hasattr(tokenizer, "lang_code_to_id") and tgt_nllb in getattr(tokenizer, "lang_code_to_id", {}):
                forced_bos_token_id = tokenizer.lang_code_to_id[tgt_nllb]
            else:
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
            )

            # 5. Decode output tokens
            translated_text = tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True,
            )[0].strip()

            elapsed_time = round(time.perf_counter() - start_time, 4)

            logger.info(
                f"Translation ({norm_src} -> {norm_tgt}) completed in {elapsed_time}s."
            )

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=norm_src,
                target_language=norm_tgt,
                processing_time_seconds=elapsed_time,
            )
        except Exception as e:
            logger.error(f"Translation execution failed: {e}")
            raise TranslationServiceError(f"Translation failed: {e}") from e
