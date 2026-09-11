import logging
import torch

logger = logging.getLogger(__name__)

# Graceful import of 'spaces' — only available inside Hugging Face ZeroGPU environment.
# Locally, we replace @spaces.GPU with a no-op pass-through decorator.
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

# Load model on CPU at import time.
# On ZeroGPU, when the decorated function is called, ZeroGPU automatically
# moves all PyTorch tensors to the assigned GPU for the duration of the call.
logger.info("Loading Whisper model (cpu)...")
try:
    from transformers import pipeline as hf_pipeline
    stt_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device="cpu",
    )
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    stt_pipeline = None


@gpu_decorator
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using the Hugging Face Whisper pipeline.
    Decorated with @spaces.GPU for Hugging Face ZeroGPU compatibility.
    Falls back to CPU when running locally.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Transcribed text string.
    """
    if stt_pipeline is None:
        raise RuntimeError("STT pipeline is not loaded. Check startup logs for errors.")

    logger.info(f"Transcribing audio: {audio_path}")
    result = stt_pipeline(audio_path)
    text = result.get("text", "").strip()
    logger.info(f"Transcription result: {text!r}")
    return text
