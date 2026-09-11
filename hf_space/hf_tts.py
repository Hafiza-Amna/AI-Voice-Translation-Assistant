import tempfile
from gtts import gTTS
import os
import uuid

def synthesize_audio(text: str, language: str) -> str:
    """
    Synthesize audio using gTTS (Google Text-to-Speech)
    Args:
        text: Text to synthesize
        language: Language code ('ur' or 'en')
    Returns:
        Path to the generated temporary audio file (.mp3)
    """
    if not text or not text.strip():
        return None
        
    # Map to gTTS expected codes
    gtts_lang_map = {"en": "en", "ur": "ur"}
    gtts_lang = gtts_lang_map.get(language, "en")
    
    # Create temp file
    temp_file = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex[:8]}.mp3")
    
    tts = gTTS(text=text, lang=gtts_lang)
    tts.save(temp_file)
    
    return temp_file
