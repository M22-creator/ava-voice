import csv

import numpy as np

from ava_voice.detect import SpeechActivity, rms_dbfs
from ava_voice.latency import LatencyTracker

SR = 24000
F = 1920
LOUD = np.full(F, 0.1, dtype=np.float32)  # -20 dBFS
QUIET = np.zeros(F, dtype=np.float32)


def test_rms_dbfs():
    assert abs(rms_dbfs(LOUD) - -20.0) < 0.01
    assert rms_dbfs(QUIET) <= -119
    assert rms_dbfs(np.zeros(0)) == -120.0


def feed(det, chunks, t0=100.0):
    out, t = [], t0
    for c in chunks:
        out += det.update(c, t, SR)
        t += F / SR
    return out


def test_start_and_end_with_hangover():
    det = SpeechActivity(-40, hangover_s=0.4)
    events = feed(det, [QUIET, LOUD, LOUD, QUIET, QUIET, QUIET, QUIET, QUIET, QUIET])
    assert [k for k, _ in events] == ["start", "end"]
    assert events[0][1] == 100.08
    assert abs(events[1][1] - (100.0 + 3 * 0.08)) < 1e-9  # end of the last loud chunk


def test_short_gap_does_not_end_speech():
    det = SpeechActivity(-40, hangover_s=0.4)
    events = feed(det, [LOUD, QUIET, QUIET, LOUD, QUIET])
    assert [k for k, _ in events] == ["start"]
    assert det.force_end()[0][0] == "end"
    assert det.force_end() == []


def test_latency_row_written(tmp_path):
    path = tmp_path / "lat.csv"
    lt = LatencyTracker(path)
    lt.on_user_start(10.0)
    lt.on_user_end(11.0)
    row = lt.on_robot_start(11.35, "s1", 1)
    assert row["latency_ms"] == 350
    lt.on_robot_end()
    lt.on_user_start(12.0)
    lt.on_user_end(13.0)
    lt.on_robot_start(13.5, "s1", 2)
    rows = list(csv.DictReader(path.open()))
    assert [r["latency_ms"] for r in rows] == ["350", "500"]
    assert rows[0]["user_speech_end_unix"] == "11.000"


def test_no_row_when_robot_starts_while_user_talks_or_without_user_speech(tmp_path):
    lt = LatencyTracker(tmp_path / "lat.csv")
    assert lt.on_robot_start(5.0, "s", 1) is None  # robot spoke first
    lt.on_robot_end()
    lt.on_user_start(6.0)
    assert lt.on_robot_start(6.5, "s", 2) is None  # overlap
    assert not (tmp_path / "lat.csv").exists()


def test_mic_activity_during_robot_speech_is_ignored_for_latency():
    lt = LatencyTracker(None)
    lt.on_user_start(1.0)
    lt.on_user_end(2.0)
    lt.on_robot_start(2.5, "s", 1)
    lt.on_user_start(2.6)  # echo of the robot's voice
    lt.on_user_end(3.0)
    lt.on_robot_end()
    assert lt.on_robot_start(4.0, "s", 2) is None


def test_user_who_keeps_talking_after_barge_in_counts():
    lt = LatencyTracker(None)
    lt.on_robot_start(1.0, "s", 1)
    lt.on_user_start(1.5)  # barge-in while robot talks: ignored...
    lt.on_robot_end(user_still_active=True)  # ...but the user is still talking
    lt.on_user_end(3.0)
    assert lt.on_robot_start(3.4, "s", 2)["latency_ms"] == 400
