"""Real Parakeet through the sidecar path (24 kHz mic stream -> 16 kHz ASR).
Skipped unless sherpa-onnx is installed and the weights exist (python -m ava_voice.asr --download,
or AVA_ASR_MODEL_DIR pointing at a folder that contains them)."""
import os
from pathlib import Path

import numpy as np
import pytest

from ava_voice.asr import PARAKEET_DIR, ParakeetASR, resample
from ava_voice.config import REPO_ROOT

MODEL_DIR = Path(os.environ.get("AVA_ASR_MODEL_DIR", REPO_ROOT / "models"))
WAV = MODEL_DIR / PARAKEET_DIR / "test_wavs" / "0.wav"


def test_parakeet_transcribes_a_24khz_stream():
    pytest.importorskip("sherpa_onnx")
    if not WAV.exists():
        pytest.skip("Parakeet weights not downloaded")
    import sphn

    audio16, sr = sphn.read(str(WAV))
    assert sr == 16000
    mic24 = resample(audio16[0].astype(np.float32), 16000, 24000)  # what the client would capture
    res = ParakeetASR(MODEL_DIR).transcribe(mic24, 24000)
    words = res["text"].lower().replace(",", "").replace(".", "").split()
    for w in ("wish", "observed", "turning", "eyes", "certainly", "portrait"):
        assert w in words, res["text"]
    assert res["asr_ms"] > 0
