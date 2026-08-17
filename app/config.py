from pathlib import Path
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings using Pydantic Settings management."""
    
    # App Settings
    APP_NAME: str = "AI Voice Translation Assistant"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Azure Speech Credentials
    AZURE_SPEECH_KEY: str = "placeholder_key"
    AZURE_SPEECH_REGION: str = "placeholder_region"
    
    # Speech-to-Text (faster-whisper)
    WHISPER_MODEL_SIZE: str = "small"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    
    # Translation (NLLB-200)
    NLLB_MODEL_NAME: str = Field(
        default="facebook/nllb-200-distilled-600M",
        validation_alias=AliasChoices("NLLB_MODEL_NAME", "NLLB_MODEL"),
    )
    NLLB_DEVICE: str = "cpu"

    # Text-to-Speech (Piper TTS — offline)
    PIPER_EXECUTABLE: str = "piper"
    PIPER_MODELS_DIR: str = "./piper_models"
    PIPER_ENGLISH_VOICE: str = "en_US-lessac-medium.onnx"
    PIPER_URDU_VOICE: str = "ur_PK-usman-medium.onnx"
    TTS_OUTPUT_DIR: str = "./generated_audio"

    # Audio Settings
    TEMP_AUDIO_DIR: str = "./audio_temp"
    MAX_AUDIO_SIZE_MB: int = 10
    
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
