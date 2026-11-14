"""Lyrics transcription worker (runs in the .venv-asr environment, not the main one).

Usage: python asr_worker.py vocals.wav out.json [--model large-v3] [--device cpu] [--compute int8] [--language es]
Writes JSON: {language, language_probability, words:[{w,start,end,p}], segments:[{start,end,text}]}
"""
import argparse
import json
import time

import av
import numpy as np
from faster_whisper import WhisperModel


def load_16k_mono(path):
    """Decode with PyAV directly (faster-whisper 1.2.1's own decoder breaks with PyAV >= 19)."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
    for f in resampler.resample(None):
        chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--language", default=None, help="omit to auto-detect")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--vad", action="store_true", help="enable the voice-activity filter (off by default for singing)")
    ap.add_argument("--prompt", default="", help="text hint, e.g. 'Letra de una canción en español.'")
    a = ap.parse_args()

    t0 = time.time()
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute, cpu_threads=a.threads)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    audio = load_16k_mono(a.audio)
    print(f"audio: {len(audio) / 16000:.1f}s", flush=True)
    # Singing is not what Whisper was trained on: its default "this is not speech" filters drop
    # most sung phrases, so they are relaxed here.
    segs, info = model.transcribe(
        audio, language=a.language, word_timestamps=True,
        vad_filter=a.vad, vad_parameters=dict(threshold=0.3, min_silence_duration_ms=500),
        beam_size=5,
        condition_on_previous_text=False,  # avoids runaway repetition on sung, repetitive text
        no_speech_threshold=None, log_prob_threshold=None, compression_ratio_threshold=3.0,
        initial_prompt=a.prompt or None,
    )
    print(f"language: {info.language} ({info.language_probability:.2f})", flush=True)
    words, segments = [], []
    for s in segs:
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        for w in s.words or []:
            words.append({"w": w.word.strip(), "start": w.start, "end": w.end, "p": w.probability})
        print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.text.strip()}", flush=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"language": info.language, "language_probability": info.language_probability,
                   "words": words, "segments": segments}, f, ensure_ascii=False, indent=1)
    print(f"done in {time.time() - t0:.0f}s, {len(words)} words", flush=True)


if __name__ == "__main__":
    main()
