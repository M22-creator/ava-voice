import numpy as np

from ava_voice.audio import PlaybackBuffer


def test_fifo_and_underrun_padding():
    b = PlaybackBuffer(1.0, sample_rate=10)
    b.push(np.arange(4, dtype=np.float32))
    b.push(np.arange(4, 7, dtype=np.float32))
    assert b.seconds() == 0.7
    assert b.pull(5).tolist() == [0, 1, 2, 3, 4]
    assert b.pull(4).tolist() == [5, 6, 0, 0]
    assert b.seconds() == 0


def test_overflow_drops_oldest():
    b = PlaybackBuffer(0.5, sample_rate=10)  # 5 samples max
    b.push(np.arange(4, dtype=np.float32))
    b.push(np.arange(10, 14, dtype=np.float32))
    assert b.dropped_samples == 3
    assert b.pull(5).tolist() == [3, 10, 11, 12, 13]
