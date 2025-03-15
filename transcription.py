import torch
import numpy as np
import gradio as gr
from transformers import pipeline
from functools import lru_cache

device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device=device)

@lru_cache(maxsize=128)
def cached_transcribe(audio_bytes, sr):
    y = np.frombuffer(audio_bytes, dtype=np.float32)

    max_duration_sec = 30  # optimal chunk duration for Whisper
    max_chunk_samples = int(sr * max_duration_sec)
    transcript_parts = []

    for start_idx in range(0, len(y), max_chunk_samples):
        chunk = y[start_idx:start_idx + max_chunk_samples]
        result = transcriber({"sampling_rate": sr, "raw": chunk})
        transcript_parts.append(result["text"].strip())

    return " ".join(transcript_parts).strip()

def transcribe(audio):
    if audio is None:
        return "", gr.Button(interactive=False)

    sr, y = audio

    # Convert stereo to mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    audio_bytes = y.tobytes()
    full_transcript = cached_transcribe(audio_bytes, sr)

    return full_transcript, gr.Button(interactive=True)
