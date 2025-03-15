import tempfile
import os
import soundfile as sf
import numpy as np
from TTS.api import TTS
from tts_cache import TTSCache
import torch

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True if torch.cuda.is_available() else False)
tts_cache = TTSCache(max_size=100)

def generate_speech(text):
    cached_result = tts_cache.get(text)
    if cached_result is not None:
        return cached_result
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        temp_filename = fp.name
    
    tts.tts_to_file(text=text, file_path=temp_filename)
    speech, sample_rate = sf.read(temp_filename, dtype='float32')
    os.unlink(temp_filename)
    
    result = (int(sample_rate), speech.copy())
    tts_cache.put(text, result)
    
    return result
