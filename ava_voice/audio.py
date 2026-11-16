"""Microphone capture and speaker playback.

The client talks to audio through the small ``AudioIO`` interface below so the
tests can swap in fakes. ``SoundDeviceIO`` is the real implementation; it
opens the devices at the model's 24 kHz rate and relies on the OS audio
server (PulseAudio/PipeWire, or ALSA's ``default``/``pulse`` devices) to
resample. Raw ``hw:`` ALSA devices often do not support 24 kHz.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable, Optional, Protocol

import numpy as np

from .protocol import FRAME_SIZE, SAMPLE_RATE

log = logging.getLogger(__name__)

# on_mic(pcm_float32_mono, capture_start_unix_time)
MicCallback = Callable[[np.ndarray, float], None]


class AudioIO(Protocol):
    def start(self, on_mic: MicCallback) -> None: ...
    def stop(self) -> None: ...
    def play(self, pcm: np.ndarray) -> None: ...
    def buffered_seconds(self) -> float: ...
    def output_latency(self) -> float: ...
    def clear_playback(self) -> None: ...


class PlaybackBuffer:
    """Thread-safe FIFO of float32 samples between the network and the speaker callback."""

    def __init__(self, max_seconds: float, sample_rate: int = SAMPLE_RATE):
        self.max_samples = int(max_seconds * sample_rate)
        self.sample_rate = sample_rate
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._size = 0
        self._lock = threading.Lock()
        self.dropped_samples = 0

    def push(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(pcm)
            self._size += pcm.size
            while self._size > self.max_samples and self._chunks:
                over = self._size - self.max_samples
                head = self._chunks[0]
                if head.size <= over:
                    self._chunks.popleft()
                    self._size -= head.size
                    self.dropped_samples += head.size
                else:
                    self._chunks[0] = head[over:]
                    self._size -= over
                    self.dropped_samples += over

    def pull(self, n: int) -> np.ndarray:
        """Return exactly n samples, zero-padded on underrun."""
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                head = self._chunks[0]
                take = min(n - filled, head.size)
                out[filled:filled + take] = head[:take]
                filled += take
                if take == head.size:
                    self._chunks.popleft()
                else:
                    self._chunks[0] = head[take:]
                self._size -= take
        return out

    def seconds(self) -> float:
        with self._lock:
            return self._size / self.sample_rate

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._size = 0


def _sd():
    import sounddevice  # imported lazily: needs PortAudio, and env vars may be set first

    return sounddevice


def list_devices() -> str:
    sd = _sd()
    lines = []
    default_in, default_out = sd.default.device
    for idx, dev in enumerate(sd.query_devices()):
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        marks = []
        if idx == default_in:
            marks.append("default input")
        if idx == default_out:
            marks.append("default output")
        lines.append(
            f"{idx:3d}  in={dev['max_input_channels']:<2d} out={dev['max_output_channels']:<2d} "
            f"{dev['default_samplerate']:>7.0f} Hz  [{hostapi}]  {dev['name']}"
            + (f"   <- {', '.join(marks)}" if marks else "")
        )
    return "\n".join(lines)


def resolve_device(spec, kind: str):
    """Config value -> sounddevice device. None = system default; int = index;
    str = case-insensitive substring of the device name (must match exactly one)."""
    if spec is None or isinstance(spec, int):
        return spec
    if isinstance(spec, str) and spec.strip().isdigit():
        return int(spec)
    sd = _sd()
    chan_key = "max_input_channels" if kind == "input" else "max_output_channels"
    usable = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d[chan_key] > 0]
    exact = [i for i, name in usable if name.lower() == spec.lower()]
    if len(exact) == 1:
        return exact[0]
    matches = [i for i, name in usable if spec.lower() in name.lower()]
    if not matches:
        raise RuntimeError(f"no {kind} device matches '{spec}'. Run ./run.sh --list-devices")
    if len(matches) > 1:
        names = ", ".join(f"{i}: {sd.query_devices(i)['name']}" for i in matches)
        raise RuntimeError(f"'{spec}' matches several {kind} devices ({names}); use an index or a longer name")
    return matches[0]


class SoundDeviceIO:
    def __init__(self, input_device=None, output_device=None, max_buffer_s: float = 1.0):
        self.input_device = input_device
        self.output_device = output_device
        self.buffer = PlaybackBuffer(max_buffer_s)
        self._in = None
        self._out = None

    def start(self, on_mic: MicCallback) -> None:
        sd = _sd()
        indev = resolve_device(self.input_device, "input")
        outdev = resolve_device(self.output_device, "output")

        def in_cb(indata, frames, time_info, status):
            if status:
                log.debug("input status: %s", status)
            latency = self._in.latency if self._in is not None else 0.0
            t0 = time.time() - frames / SAMPLE_RATE - latency
            on_mic(indata[:, 0].copy(), t0)

        def out_cb(outdata, frames, time_info, status):
            if status:
                log.debug("output status: %s", status)
            outdata[:, 0] = self.buffer.pull(frames)

        try:
            self._out = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SIZE,
                device=outdev, callback=out_cb, latency="low",
            )
            self._in = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SIZE,
                device=indev, callback=in_cb, latency="low",
            )
        except sd.PortAudioError as e:
            raise RuntimeError(
                f"cannot open audio devices at {SAMPLE_RATE} Hz mono ({e}). Raw ALSA 'hw:' devices often "
                "do not support 24 kHz; choose the 'pulse', 'pipewire' or 'default' device instead "
                "(see ./run.sh --list-devices)."
            ) from e
        self._out.start()
        self._in.start()
        log.info("audio: input=%s output=%s", self._in.device, self._out.device)

    def stop(self) -> None:
        for s in (self._in, self._out):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception as e:  # noqa: BLE001 - best effort on shutdown
                    log.debug("closing stream: %s", e)
        self._in = self._out = None

    def play(self, pcm: np.ndarray) -> None:
        self.buffer.push(pcm)

    def buffered_seconds(self) -> float:
        return self.buffer.seconds()

    def output_latency(self) -> float:
        return float(self._out.latency) if self._out is not None else 0.0

    def clear_playback(self) -> None:
        self.buffer.clear()
