#!/usr/bin/env python3
"""Can this machine run PersonaPlex-7B in real time?

Prints the GPU and its VRAM, runs one clip through upstream's offline
inference (the same encode -> LM step -> decode loop the server runs for
every 80 ms frame), and reports peak GPU memory and the real-time factor.

    .venv/bin/python scripts/check_hardware.py [--input-wav clip.wav] [--cpu-offload]

Exit codes: 0 = real time (possibly with a warning), 1 = too slow or failed,
2 = no usable CUDA GPU / PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = ROOT / "third_party/personaplex/assets/test/input_assistant.wav"
FRAME_MS = 80.0  # 12.5 frames per second
MIN_VRAM_GB = 24.0
RTF_WARN = 0.8


def assess(rtf: float, p95_ms: float, frame_ms: float = FRAME_MS) -> tuple[bool, str]:
    """Return (ok, explanation). RTF = processing time / audio duration."""
    if rtf >= 1.0:
        return False, (
            f"Too slow for real time: each 80 ms of audio took {rtf * frame_ms:.0f} ms on average "
            f"(real-time factor {rtf:.2f}, must be below 1.0). The robot's replies would fall further "
            "and further behind. Use a faster GPU; --cpu-offload is not expected to be real time."
        )
    if rtf > RTF_WARN or p95_ms > frame_ms:
        return True, (
            f"Real time, but with little headroom (RTF {rtf:.2f}, 95th percentile frame {p95_ms:.0f} ms "
            f"vs {frame_ms:.0f} ms budget). Expect occasional stutter, especially if other GPU work runs."
        )
    return True, f"Real time with headroom (RTF {rtf:.2f}, 95th percentile frame {p95_ms:.0f} ms)."


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    k = min(len(s) - 1, max(0, round(q / 100.0 * (len(s) - 1))))
    return s[k]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-wav", type=Path, default=DEFAULT_WAV)
    p.add_argument("--voice", default="NATF2")
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--seed", type=int, default=42424242)
    args = p.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this Python. Run ./setup.sh and use .venv/bin/python.")
        return 2

    if not torch.cuda.is_available():
        print(
            "No CUDA GPU is visible to PyTorch, so PersonaPlex cannot run in real time here.\n"
            "Check `nvidia-smi`; if it works, PyTorch may be a CPU-only build or too old for the GPU."
        )
        return 2

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"GPU:        {props.name}")
    print(f"VRAM:       {vram_gb:.1f} GiB")
    print(f"PyTorch:    {torch.__version__} (CUDA {torch.version.cuda})")
    if vram_gb < MIN_VRAM_GB and not args.cpu_offload:
        print(f"WARNING: less than {MIN_VRAM_GB:.0f} GiB VRAM; loading the model may run out of memory.")

    if not args.input_wav.exists():
        print(f"Input clip not found: {args.input_wav} (run ./setup.sh, or pass --input-wav).")
        return 1

    try:
        import moshi.offline as offline
    except ImportError as e:
        print(f"Cannot import upstream moshi ({e}). Run ./setup.sh first.")
        return 2

    # Time the streaming loop only (not model loading): wrap the generator that
    # offline.run_inference iterates once per 80 ms frame, and record when each
    # frame is requested. The loop body syncs with the GPU (.item() / .cpu()),
    # so these wall-clock gaps are real per-frame processing times.
    stamps: list[float] = []
    original = offline.lm_encode_from_sphn

    def timed_encode(*a, **kw):
        for item in original(*a, **kw):
            stamps.append(time.perf_counter())
            yield item
        stamps.append(time.perf_counter())

    offline.lm_encode_from_sphn = timed_encode

    voice_dir = offline._get_voice_prompt_dir(None, offline.loaders.DEFAULT_REPO)
    voice_path = os.path.join(voice_dir, f"{args.voice}.pt")
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory() as tmp:
        out_wav, out_json = os.path.join(tmp, "out.wav"), os.path.join(tmp, "out.json")
        print(f"\nRunning one offline clip: {args.input_wav.name} (voice {args.voice}) ...", flush=True)
        try:
            with torch.no_grad():
                offline.run_inference(
                    input_wav=str(args.input_wav), output_wav=out_wav, output_text=out_json,
                    text_prompt="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
                    voice_prompt_path=voice_path, tokenizer_path=None, moshi_weight=None, mimi_weight=None,
                    hf_repo=offline.loaders.DEFAULT_REPO, device="cuda", seed=args.seed,
                    temp_audio=0.8, temp_text=0.7, topk_audio=250, topk_text=25, greedy=False,
                    save_voice_prompt_embeddings=False, cpu_offload=args.cpu_offload,
                )
        except torch.cuda.OutOfMemoryError:
            print(f"Out of GPU memory: {props.name} ({vram_gb:.1f} GiB) cannot hold the model. "
                  "Try --cpu-offload (not expected to be real time) or a GPU with >= 24 GiB.")
            return 1
        finally:
            offline.lm_encode_from_sphn = original
        reply = ""
        if os.path.exists(out_json):
            tokens = json.load(open(out_json))
            reply = "".join(t for t in tokens if t not in ("EPAD", "BOS", "EOS", "PAD")).strip()

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    frame_s = [b - a for a, b in zip(stamps, stamps[1:])][1:]  # drop the first (CUDA warm-up effects)
    if not frame_s:
        print("The clip produced no frames; cannot measure speed.")
        return 1
    audio_s = len(frame_s) * FRAME_MS / 1000.0
    rtf = sum(frame_s) / audio_s
    p95 = percentile([s * 1000 for s in frame_s], 95)

    print(f"\nModel reply text: {reply[:200]!r}")
    print(f"Frames:     {len(frame_s)} ({audio_s:.1f} s of audio)")
    print(f"Per frame:  mean {1000 * sum(frame_s) / len(frame_s):.1f} ms, p95 {p95:.1f} ms (budget {FRAME_MS:.0f} ms)")
    print(f"RTF:        {rtf:.2f}   (processing time / audio time; < 1.0 is real time)")
    print(f"Peak GPU memory: {peak_alloc:.1f} GiB allocated, {peak_reserved:.1f} GiB reserved by PyTorch")
    ok, why = assess(rtf, p95)
    print(("\nOK: " if ok else "\nFAIL: ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
