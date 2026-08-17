"""Unit tests for TranslationService using mocked Hugging Face models."""

from unittest.mock import MagicMock, patch
import pytest

from app.services.translation_service import (
    TranslationService,
    TranslationServiceError,
    UnsupportedLanguageError,
)
from app.models.schemas import TranslationResult


@pytest.fixture
def mock_transformers():
    """Fixture to mock AutoTokenizer and AutoModelForSeq2SeqLM."""
    with patch("app.services.translation_service.AutoTokenizer") as mock_tok_cls, \
         patch("app.services.translation_service.AutoModelForSeq2SeqLM") as mock_mod_cls:
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 12345
        mock_tokenizer.lang_code_to_id = {"eng_Latn": 12345, "urd_Arab": 54321}
        mock_tokenizer.batch_decode.return_value = ["Hello, how are you?"]
        
        # Mock encoding return object
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_model.to.return_value = mock_model

        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_mod_cls.from_pretrained.return_value = mock_model

        yield {
            "tok_cls": mock_tok_cls,
            "mod_cls": mock_mod_cls,
            "tokenizer": mock_tokenizer,
            "model": mock_model,
        }


def test_urdu_to_english_translation(mock_transformers):
    """Test translating Urdu text to English."""
    service = TranslationService()
    result = service.translate("السلام علیکم", source_lang="ur", target_lang="en")

    assert isinstance(result, TranslationResult)
    assert result.original_text == "السلام علیکم"
    assert result.translated_text == "Hello, how are you?"
    assert result.source_language == "ur"
    assert result.target_language == "en"
    assert result.processing_time_seconds >= 0.0


def test_english_to_urdu_translation(mock_transformers):
    """Test translating English text to Urdu."""
    mock_transformers["tokenizer"].batch_decode.return_value = ["السلام علیکم"]
    
    service = TranslationService()
    result = service.translate("Hello, how are you?", source_lang="en", target_lang="ur")

    assert isinstance(result, TranslationResult)
    assert result.original_text == "Hello, how are you?"
    assert result.translated_text == "السلام علیکم"
    assert result.source_language == "en"
    assert result.target_language == "ur"
    assert result.processing_time_seconds >= 0.0


def test_empty_text_handling(mock_transformers):
    """Test graceful handling of empty or whitespace text."""
    service = TranslationService()
    result = service.translate("   ", source_lang="ur", target_lang="en")

    assert result.original_text == "   "
    assert result.translated_text == ""
    assert result.source_language == "ur"
    assert result.target_language == "en"
    assert result.processing_time_seconds == 0.0

    # Ensure model generation was skipped for empty text
    mock_transformers["model"].generate.assert_not_called()


def test_unsupported_source_language():
    """Test handling of unsupported source language code."""
    service = TranslationService()
    with pytest.raises(UnsupportedLanguageError):
        service.translate("Hello", source_lang="fr", target_lang="en")


def test_unsupported_target_language():
    """Test handling of unsupported target language code."""
    service = TranslationService()
    with pytest.raises(UnsupportedLanguageError):
        service.translate("Hello", source_lang="en", target_lang="es")


def test_model_initialization(mock_transformers):
    """Test model and tokenizer initialization with config settings."""
    service = TranslationService(model_name="custom/model-name", device="cpu")
    tokenizer, model = service.load_model()

    mock_transformers["tok_cls"].from_pretrained.assert_called_once_with("custom/model-name")
    mock_transformers["mod_cls"].from_pretrained.assert_called_once_with("custom/model-name")
    assert tokenizer == mock_transformers["tokenizer"]
    assert model == mock_transformers["model"]


def test_model_loaded_only_once(mock_transformers):
    """Test that model is loaded only once and reused across multiple translate calls."""
    service = TranslationService()

    service.translate("First text", source_lang="en", target_lang="ur")
    service.translate("Second text", source_lang="ur", target_lang="en")

    assert mock_transformers["tok_cls"].from_pretrained.call_count == 1
    assert mock_transformers["mod_cls"].from_pretrained.call_count == 1


def test_model_initialization_failure():
    """Test handling of exception during model loading."""
    with patch("app.services.translation_service.AutoTokenizer.from_pretrained", side_effect=Exception("Network error")):
        service = TranslationService()
        with pytest.raises(TranslationServiceError) as exc_info:
            service.load_model()
        assert "Model initialization failed" in str(exc_info.value)
