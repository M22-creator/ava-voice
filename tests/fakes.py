"""Fake audio devices for tests."""

from __future__ import annotations

import threading
import time

import numpy as np

from ava_voice.audio import PlaybackBuffer

SR = 24000
FRAME = 1920


class FakeAudio:
    """Mic plays a script of (seconds, amplitude) segments in real time, then silence.
    Speaker output is collected in ``played``."""

    def __init__(self, script=((0.5, 0.3), (0.0, 0.0))):
        self.script = script
        self.played: list[np.ndarray] = []
        self.buffer = PlaybackBuffer(1.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started = self.stopped = False

    def _mic_samples(self):
        for seconds, amp in self.script:
            n = int(seconds * SR)
            t = np.arange(n) / SR
            yield from np.array_split((amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32), max(1, n // FRAME))
        while True:
            yield np.zeros(FRAME, dtype=np.float32)

    def start(self, on_mic) -> None:
        self.started = True

        def run():
            for chunk in self._mic_samples():
                if self._stop.is_set():
                    return
                if chunk.size == 0:
                    continue
                on_mic(chunk, time.time())
                time.sleep(chunk.size / SR)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stopped = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def play(self, pcm) -> None:
        self.played.append(pcm)

    def buffered_seconds(self) -> float:
        return 0.0

    def output_latency(self) -> float:
        return 0.0

    def clear_playback(self) -> None:
        pass
