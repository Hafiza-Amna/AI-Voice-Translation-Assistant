"""Language utilities for normalization and mapping."""

SUPPORTED_LANGUAGES = {
    "ur": "Urdu",
    "en": "English",
}

# Alias mapping for variations in language codes
LANGUAGE_ALIASES = {
    "urdu": "ur",
    "english": "en",
    "urd_arab": "ur",
    "eng_latn": "en",
}

# NLLB-200 specific language code tokens
NLLB_LANGUAGE_MAP = {
    "ur": "urd_Arab",
    "en": "eng_Latn",
}

def normalize_language_code(lang: str) -> str:
    """
    Normalizes input language string to standard ISO language code.
    Defaults to returning lowercase string if not specifically mapped.
    """
    if not lang:
        return ""
    code = lang.strip().lower()
    return LANGUAGE_ALIASES.get(code, code)

def is_supported_language(lang_code: str) -> bool:
    """Checks if normalized language code is supported."""
    code = normalize_language_code(lang_code)
    return code in SUPPORTED_LANGUAGES

def get_nllb_code(lang: str) -> str:
    """
    Maps language input to NLLB-200 token (e.g. 'ur' -> 'urd_Arab', 'en' -> 'eng_Latn').
    
    Returns:
        str: NLLB language token string.

    Raises:
        ValueError: If language is not supported.
    """
    normalized = normalize_language_code(lang)
    if normalized in NLLB_LANGUAGE_MAP:
        return NLLB_LANGUAGE_MAP[normalized]
    raise ValueError(f"Unsupported language code for translation: '{lang}'")


def get_piper_voice(lang: str, english_voice: str, urdu_voice: str) -> str:
    """
    Returns the Piper TTS voice model filename for the given language code.

    Args:
        lang: Language code input (e.g., 'ur', 'en', 'urdu', 'english').
        english_voice: Configured English voice model filename.
        urdu_voice: Configured Urdu voice model filename.

    Returns:
        str: Voice model filename (e.g., 'en_US-lessac-medium.onnx').

    Raises:
        ValueError: If language is not supported for TTS.
    """
    normalized = normalize_language_code(lang)
    voice_map = {
        "en": english_voice,
        "ur": urdu_voice,
    }
    if normalized in voice_map:
        return voice_map[normalized]
    raise ValueError(f"Unsupported language code for TTS: '{lang}'")
