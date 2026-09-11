import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

# Graceful import of 'spaces' — only available inside Hugging Face ZeroGPU environment.
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Maps simple ISO codes to NLLB language tokens
NLLB_LANG_MAP = {
    "ur": "urd_Arab",
    "en": "eng_Latn",
}

# Load model on CPU at import time.
# ZeroGPU will move tensors to GPU inside the decorated function automatically.
logger.info(f"Loading NLLB-200 model '{MODEL_NAME}' on cpu...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    logger.info("NLLB-200 model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading NLLB-200 model: {e}")
    tokenizer = None
    model = None


@gpu_decorator
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text between Urdu ('ur') and English ('en') using NLLB-200.
    Decorated with @spaces.GPU for Hugging Face ZeroGPU compatibility.
    Falls back to CPU inference when running locally.

    Args:
        text: Input text to translate.
        source_lang: Source language code ('ur' or 'en').
        target_lang: Target language code ('ur' or 'en').

    Returns:
        Translated text string.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Translation model is not loaded. Check startup logs.")

    if not text or not text.strip():
        return ""

    src_nllb = NLLB_LANG_MAP.get(source_lang)
    tgt_nllb = NLLB_LANG_MAP.get(target_lang)

    if not src_nllb or not tgt_nllb:
        raise ValueError(f"Unsupported language pair: '{source_lang}' -> '{target_lang}'")

    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt")

    # Resolve forced BOS token — supports both old and new tokenizer API
    if hasattr(tokenizer, "lang_code_to_id") and tgt_nllb in getattr(tokenizer, "lang_code_to_id", {}):
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_nllb]
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=512,
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    logger.info(f"Translation ({source_lang} -> {target_lang}): {translated_text!r}")
    return translated_text
