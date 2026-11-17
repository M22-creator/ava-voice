"""Optional transcript of what the USER said, from a separate speech recognizer.

PersonaPlex is speech-to-speech and never transcribes the user. When
config `asr.enabled` is true, the client cuts the microphone stream into
utterances with its speech detector and transcribes each one here, emitting a
`user_text` event. This text comes from NVIDIA Parakeet-TDT 0.6B v2 (chosen in
the ava-speech-eval comparison), not from PersonaPlex.

    .venv/bin/pip install sherpa-onnx
    .venv/bin/python -m ava_voice.asr --download        # ~0.5 GB into models/ (git-ignored)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import math
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np

ASR_RATE = 16000
PARAKEET_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PARAKEET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
PARAKEET_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
                f"{PARAKEET_DIR}.tar.bz2")
PARAKEET_SHA256 = "6fc4426b54aa549a0019b8a75b8dea59164e71790a37ac56a045ad57f915f714"
ENGINE_LABEL = "Parakeet-TDT 0.6B v2"


def resample(audio: np.ndarray, src: int, dst: int = ASR_RATE) -> np.ndarray:
    """Band-limited enough for speech: low-pass by averaging, then linear interpolation."""
    if src == dst:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst / src))
    if n <= 0:
        return np.zeros(0, np.float32)
    if dst < src:  # simple anti-alias: moving average over the decimation ratio
        k = max(1, int(round(src / dst)))
        audio = np.convolve(audio, np.ones(k, np.float32) / k, mode="same")
    x_new = np.linspace(0, len(audio) - 1, n)
    return np.interp(x_new, np.arange(len(audio)), audio).astype(np.float32)


class UtteranceCollector:
    """Feeds on fixed-size mic frames plus the speech detector's start/end events and
    returns one finished utterance (with a short pre-roll) when the detector says it ended."""

    def __init__(self, frame_s: float, preroll_s: float = 0.3, max_s: float = 20.0) -> None:
        self.preroll: Deque[np.ndarray] = collections.deque(maxlen=max(1, math.ceil(preroll_s / frame_s)))
        self.max_frames = max(1, int(max_s / frame_s))
        self.buf: Optional[List[np.ndarray]] = None

    def feed(self, frame: np.ndarray, events: List[tuple]) -> Optional[np.ndarray]:
        kinds = [k for k, _ in events]
        out = None
        if "start" in kinds and self.buf is None:
            self.buf = list(self.preroll)
        if self.buf is not None:
            self.buf.append(frame)
            if "end" in kinds or len(self.buf) >= self.max_frames:
                out, self.buf = np.concatenate(self.buf), None
        self.preroll.append(frame)
        return out


class ParakeetASR:
    label = ENGINE_LABEL

    def __init__(self, model_dir: Path, threads: int = 2) -> None:
        import sherpa_onnx

        d = Path(model_dir) / PARAKEET_DIR
        missing = [f for f in PARAKEET_FILES if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {d}. Run: .venv/bin/python -m ava_voice.asr --download")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), tokens=str(d / "tokens.txt"),
            num_threads=threads, sample_rate=ASR_RATE, feature_dim=80,
            decoding_method="greedy_search", model_type="nemo_transducer",
        )

    def transcribe(self, audio: np.ndarray, rate: int) -> dict:
        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(ASR_RATE, resample(audio, rate))
        self.recognizer.decode_stream(stream)
        return {"text": stream.result.text.strip(), "asr_ms": round((time.perf_counter() - t0) * 1000, 1)}


def build_asr(cfg, base_dir: Path):
    """cfg: AsrConfig. Returns None when disabled."""
    if not cfg.enabled:
        return None
    model_dir = Path(cfg.model_dir)
    return ParakeetASR(model_dir if model_dir.is_absolute() else base_dir / model_dir, cfg.threads)


def download(model_dir: Path) -> None:
    if all((model_dir / PARAKEET_DIR / f).exists() for f in PARAKEET_FILES):
        print(f"ok       {model_dir / PARAKEET_DIR}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = model_dir / (PARAKEET_DIR + ".tar.bz2.part")
    print(f"download {PARAKEET_URL}", flush=True)
    try:
        urllib.request.urlretrieve(PARAKEET_URL, tmp)
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != PARAKEET_SHA256:
            raise SystemExit("Checksum mismatch for the Parakeet model; file discarded.")
        with tarfile.open(tmp) as t:
            root = model_dir.resolve()
            for m in t.getmembers():
                p = (model_dir / m.name).resolve()
                if m.issym() or m.islnk() or (p != root and root not in p.parents):
                    raise SystemExit(f"Refusing unsafe archive member {m.name}")
            t.extractall(model_dir, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"saved    {model_dir / PARAKEET_DIR}")


def main() -> None:
    from .config import REPO_ROOT

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true")
    p.add_argument("--dir", type=Path, default=REPO_ROOT / "models")
    a = p.parse_args()
    if not a.download:
        p.error("nothing to do (use --download)")
    download(a.dir)


if __name__ == "__main__":
    main()
