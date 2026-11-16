"""Energy-based speech activity detection.

PersonaPlex streams audio continuously (near-silence when the robot is not
talking), so "the robot started/stopped speaking" has to be inferred from the
level of the audio it sends. The same detector, on the mic signal, gives the
"user stopped speaking" timestamp for latency logging. It is deliberately
simple: a level threshold plus a hangover time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def rms_dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))


class SpeechActivity:
    """Feed consecutive audio chunks with their start time; get start/end transitions.

    ``update`` returns a list of ``("start", t)`` / ``("end", t)`` tuples. The
    start time is the start of the first loud chunk; the end time is the end of
    the last loud chunk, reported once ``hangover_s`` of quiet audio has
    followed it.
    """

    def __init__(self, threshold_dbfs: float, hangover_s: float):
        self.threshold_dbfs = threshold_dbfs
        self.hangover_s = hangover_s
        self.active = False
        self.last_loud_end: Optional[float] = None

    def update(self, pcm: np.ndarray, t_start: float, sample_rate: int) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        t_end = t_start + pcm.shape[-1] / sample_rate
        if rms_dbfs(pcm) >= self.threshold_dbfs:
            if not self.active:
                self.active = True
                out.append(("start", t_start))
            self.last_loud_end = t_end
        elif self.active and self.last_loud_end is not None and t_end - self.last_loud_end >= self.hangover_s:
            self.active = False
            out.append(("end", self.last_loud_end))
        return out

    def force_end(self) -> list[tuple[str, float]]:
        if self.active:
            self.active = False
            return [("end", self.last_loud_end or 0.0)]
        return []
