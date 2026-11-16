"""Response latency logging: end of user speech -> first audible robot audio."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

CSV_COLUMNS = ["session", "turn", "user_speech_end_unix", "robot_audio_start_unix", "latency_ms"]


class LatencyTracker:
    """Pairs each robot turn with the user speech that ended just before it.

    A row is written when the robot starts a turn and the user's last speech
    segment ended after the previous robot turn started and before this one.
    Mic activity *while the robot is speaking* is ignored for this metric only
    (without echo cancellation it is mostly the robot's own voice); the mic
    audio itself is still streamed to the model, so interruption still works.
    """

    def __init__(self, csv_path: Optional[Path]):
        self.csv_path = Path(csv_path) if csv_path else None
        self.robot_speaking = False
        self.user_speaking = False
        self._pending_user_end: Optional[float] = None
        self.rows: list[dict] = []

    def on_user_start(self, t: float) -> None:
        if not self.robot_speaking:
            self.user_speaking = True
            self._pending_user_end = None

    def on_user_end(self, t: float) -> None:
        if self.user_speaking:
            self.user_speaking = False
            self._pending_user_end = t

    def on_robot_start(self, t_audible: float, session: Optional[str], turn: int) -> Optional[dict]:
        self.robot_speaking = True
        user_end, self._pending_user_end = self._pending_user_end, None
        # Robot started while the user was still talking (overlap/backchannel):
        # there is no "end of user speech" to measure from.
        if self.user_speaking or user_end is None or user_end > t_audible:
            self.user_speaking = False
            return None
        row = {
            "session": session,
            "turn": turn,
            "user_speech_end_unix": f"{user_end:.3f}",
            "robot_audio_start_unix": f"{t_audible:.3f}",
            "latency_ms": round((t_audible - user_end) * 1000.0),
        }
        self.rows.append(row)
        self._write(row)
        return row

    def on_robot_end(self, user_still_active: bool = False) -> None:
        """``user_still_active``: the mic detector is still in a speech segment
        (e.g. the user interrupted and kept talking); its end will count."""
        self.robot_speaking = False
        if user_still_active:
            self.user_speaking = True
            self._pending_user_end = None

    def _write(self, row: dict) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)
