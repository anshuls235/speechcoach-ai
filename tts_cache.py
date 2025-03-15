from collections import OrderedDict
import numpy as np
import io

class TTSCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, text):
        if text not in self.cache:
            self.misses += 1
            return None
        
        # Move the accessed item to the end (mark as most recently used)
        value = self.cache.pop(text)
        self.cache[text] = value
        self.hits += 1
        
        # Deserialize the cached value with explicit dtype preservation
        sample_rate, audio_dtype, audio_bytes = value
        audio_buffer = io.BytesIO(audio_bytes)
        audio = np.load(audio_buffer, allow_pickle=False)
        
        # Ensure correct dtype is restored
        if audio_dtype != str(audio.dtype):
            audio = audio.astype(np.dtype(audio_dtype))
        
        return (sample_rate, audio.copy())
    
    def put(self, text, value):
        # If text exists, remove it
        if text in self.cache:
            self.cache.pop(text)
        
        # If cache is full, remove the least recently used item (first item)
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        # Explicitly preserve dtype along with data
        sample_rate, audio = value
        audio_dtype = str(audio.dtype)
        
        # Serialize with highest precision
        audio_bytes = io.BytesIO()
        np.save(audio_bytes, audio, allow_pickle=False)
        audio_bytes = audio_bytes.getvalue()
        
        # Store serialized version with dtype information
        self.cache[text] = (sample_rate, audio_dtype, audio_bytes)
    
    def stats(self):
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }