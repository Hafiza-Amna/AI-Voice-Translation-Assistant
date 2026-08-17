"""Translation API router endpoints."""

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import (
    get_stt_service,
    get_translation_service,
    get_tts_service,
)
from app.config import settings
from app.models.schemas import AudioTranslationResponse, TextTranslationRequest, TextTranslationResponse
from app.services.stt_service import STTService, STTServiceError
from app.services.translation_service import (
    TranslationService,
    TranslationServiceError,
)
from app.services.tts_service import TTSService, TTSServiceError
from app.utils.audio import AudioValidationError, validate_audio_file
from app.utils.language import is_supported_language, normalize_language_code

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/translate-audio",
    response_model=AudioTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate audio file between Urdu and English",
    description="Full pipeline: Speech-to-Text -> Text Translation -> Text-to-Speech",
)
async def translate_audio(
    audio_file: Annotated[UploadFile, File(description="Audio file to translate (.wav, .mp3, .m4a, .ogg)")],
    source_language: Annotated[str, Form(description="Source language code ('ur' or 'en')")],
    target_language: Annotated[str, Form(description="Target language code ('ur' or 'en')")],
    stt_service: STTService = Depends(get_stt_service),
    translation_service: TranslationService = Depends(get_translation_service),
    tts_service: TTSService = Depends(get_tts_service),
) -> AudioTranslationResponse:
    """
    Translates spoken speech from an input audio file into translated audio in the target language.
    """
    start_time = time.perf_counter()

    # 1. Validate language parameters
    norm_source = normalize_language_code(source_language)
    norm_target = normalize_language_code(target_language)

    if not is_supported_language(norm_source):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported source language: '{source_language}'. Supported languages are Urdu ('ur') and English ('en').",
        )

    if not is_supported_language(norm_target):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported target language: '{target_language}'. Supported languages are Urdu ('ur') and English ('en').",
        )

    # 2. Check uploaded file presence
    if not audio_file or not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty or missing audio file upload.",
        )

    # Save uploaded file to temp directory for processing
    temp_dir = Path(settings.TEMP_AUDIO_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_ext = Path(audio_file.filename).suffix.lower()
    temp_file_path = temp_dir / f"upload_{uuid.uuid4().hex[:8]}{file_ext}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # For debugging purposes, copy the file to a permanent predictable location
        debug_path = Path("tests/data/last_recording.wav")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_file_path, debug_path)
        logger.info(f"DEBUG: Saved incoming audio to {debug_path}")

        # Validate file size, presence, and extension using existing audio utility
        validate_audio_file(temp_file_path)

    except AudioValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded audio file.",
        )

    try:
        # Step A: Speech-to-Text
        try:
            stt_result = stt_service.transcribe(
                audio_path=str(temp_file_path),
            )
            if not stt_result.text or not stt_result.text.strip():
                logger.error("STT returned empty text for the provided audio.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Speech recognition failed: No speech detected in the audio file. Please speak more clearly or check your microphone.",
                )
        except STTServiceError as e:
            logger.error(f"STT transcription failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech-to-Text transcription failed: {e}",
            )
        except Exception as e:
            logger.error(f"Unexpected STT error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected transcription error: {e}",
            )

        # Step B: Translation
        try:
            translation_result = translation_service.translate(
                text=stt_result.text,
                source_lang=norm_source,
                target_lang=norm_target,
            )
        except TranslationServiceError as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text translation failed: {e}",
            )

        # Step C: Text-to-Speech
        try:
            tts_result = tts_service.synthesize(
                text=translation_result.translated_text,
                language=norm_target,
            )
        except TTSServiceError as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text-to-Speech synthesis failed: {e}",
            )

        # Step D: Verify generated audio file existence
        generated_path = Path(tts_result.audio_path) if tts_result.audio_path else None
        if generated_path and not generated_path.exists():
            logger.error(f"Generated audio file missing: {tts_result.audio_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Generated audio file not found at path '{tts_result.audio_path}'.",
            )

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return AudioTranslationResponse(
            success=True,
            transcription=stt_result.text,
            translation=translation_result.translated_text,
            source_language=norm_source,
            target_language=norm_target,
            audio_file=tts_result.audio_path,
            processing_time=elapsed_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Catch-all: ensures no unhandled exception returns an HTML 500 page
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline error: {e}",
        )
    finally:
        # Cleanup input temporary file safely
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp audio file {temp_file_path}: {e}")

@router.post(
    "/translate-text",
    response_model=TextTranslationResponse,
    status_code=status.HTTP_200_OK,
    summary="Translate text between Urdu and English and synthesize speech",
    description="Full pipeline: Text Translation -> Text-to-Speech",
)
async def translate_text(
    request: TextTranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service),
    tts_service: TTSService = Depends(get_tts_service),
) -> TextTranslationResponse:
    """
    Translates input text into translated text and synthesized audio in the target language.
    """
    start_time = time.perf_counter()

    norm_source = normalize_language_code(request.source_language)
    norm_target = normalize_language_code(request.target_language)

    if not is_supported_language(norm_source):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported source language: '{request.source_language}'. Supported languages are Urdu ('ur') and English ('en').",
        )

    if not is_supported_language(norm_target):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported target language: '{request.target_language}'. Supported languages are Urdu ('ur') and English ('en').",
        )

    try:
        # Step A: Translation
        try:
            translation_result = translation_service.translate(
                text=request.text,
                source_lang=norm_source,
                target_lang=norm_target,
            )
        except TranslationServiceError as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text translation failed: {e}",
            )

        # Step B: Text-to-Speech
        try:
            tts_result = tts_service.synthesize(
                text=translation_result.translated_text,
                language=norm_target,
            )
        except TTSServiceError as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Text-to-Speech synthesis failed: {e}",
            )

        generated_path = Path(tts_result.audio_path) if tts_result.audio_path else None
        if generated_path and not generated_path.exists():
            logger.error(f"Generated audio file missing: {tts_result.audio_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Generated audio file not found at path '{tts_result.audio_path}'.",
            )

        elapsed_time = round(time.perf_counter() - start_time, 4)

        return TextTranslationResponse(
            success=True,
            translation=translation_result.translated_text,
            source_language=norm_source,
            target_language=norm_target,
            audio_file=tts_result.audio_path,
            processing_time=elapsed_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline error: {e}",
        )
