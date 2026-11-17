"""Live page, ASR sidecar plumbing, and the user_text event (no GPU, no real ASR needed)."""
import asyncio
import io
import json

import aiohttp
import numpy as np
import pytest

from ava_voice import protocol
from ava_voice.asr import UtteranceCollector, resample
from ava_voice.client import VoiceClient
from ava_voice.config import DetectionConfig
from ava_voice.events import EventBus
from ava_voice.latency import LatencyTracker
from ava_voice.web import WebView
from fakes import FakeAudio
from mock_server import MockServer


def test_resample_keeps_duration_and_pitch():
    t = np.arange(24000) / 24000
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = resample(tone, 24000, 16000)
    assert len(out) == 16000
    peak_hz = np.argmax(np.abs(np.fft.rfft(out))) * 16000 / len(out)
    assert abs(peak_hz - 440) < 5


def test_collector_returns_utterance_with_preroll():
    c = UtteranceCollector(frame_s=0.08, preroll_s=0.16)
    f = lambda v: np.full(1920, v, np.float32)
    assert c.feed(f(1), []) is None
    assert c.feed(f(2), []) is None
    assert c.feed(f(3), [("start", 0.0)]) is None
    assert c.feed(f(4), []) is None
    out = c.feed(f(5), [("end", 1.0)])
    assert [int(v) for v in out[::1920]] == [1, 2, 3, 4, 5]  # 2 pre-roll frames + utterance
    assert c.buf is None


def test_collector_cuts_overlong_speech():
    c = UtteranceCollector(frame_s=0.08, max_s=0.4)
    c.feed(np.zeros(1920, np.float32), [("start", 0.0)])
    outs = [c.feed(np.zeros(1920, np.float32), []) for _ in range(10)]
    assert sum(o is not None for o in outs) == 1


class FakeASR:
    label = "FakeASR"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, rate):
        self.calls.append((len(audio), rate))
        return {"text": "hello robot", "asr_ms": 12.0}


def test_user_text_event_from_the_sidecar(tmp_path):
    async def go():
        srv = MockServer()
        port = await srv.start()
        out = io.StringIO()
        asr = FakeASR()
        client = VoiceClient(protocol.build_url("127.0.0.1", port, "NATF2", "x"), FakeAudio(script=((0.6, 0.3),)),
                             EventBus(stdout=out), LatencyTracker(None), DetectionConfig(), asr=asr)
        await asyncio.wait_for(client.run(asyncio.Event(), once=True), timeout=20)
        await srv.stop()
        return asr, [json.loads(line) for line in out.getvalue().splitlines()]

    asr, events = asyncio.run(go())
    texts = [e for e in events if e["type"] == "user_text"]
    assert len(texts) == 1 and texts[0]["text"] == "hello robot" and texts[0]["engine"] == "FakeASR"
    assert 0.5 < texts[0]["audio_s"] < 1.5 and asr.calls[0][1] == 24000
    types = [e["type"] for e in events]
    assert types.index("user_text") < types.index("session_closed")
    assert srv_audio_unchanged(events)


def srv_audio_unchanged(events):
    # The sidecar must not change the conversation itself: the usual events still happen.
    types = [e["type"] for e in events]
    return "robot_speaking_start" in types and "robot_speaking_end" in types


def test_webview_history_sse_latency_and_page(tmp_path):
    async def go():
        csv = tmp_path / "lat.csv"
        csv.write_text("session,turn,user_speech_end_unix,robot_audio_start_unix,latency_ms\ns,1,10.000,10.500,500\n")
        bus = EventBus(stdout=None, session="s1")
        wv = WebView(bus, csv, port=0, history=5, info={"voice": "NATF2", "asr": None})
        await wv.start()
        base = f"http://127.0.0.1:{wv.port}"
        for i in range(7):
            bus.emit("robot_text", turn=1, delta=f" w{i}")
        async with aiohttp.ClientSession() as s:
            async with s.get(base + "/") as r:
                assert r.status == 200 and "AVA conversation" in await r.text()
            async with s.get(base + "/api/history") as r:
                h = await r.json()
            async with s.get(base + "/api/latency") as r:
                lat = await r.json()
            async with s.get(base + "/events") as r:
                await asyncio.sleep(0.1)
                bus.emit("listening", reason="robot_finished")
                line = b""
                while not line.startswith(b"data: "):
                    line = await asyncio.wait_for(r.content.readline(), 5)
        await wv.close()
        return h, lat, json.loads(line[6:])

    h, lat, ev = asyncio.run(go())
    assert [e["delta"] for e in h["events"]] == [" w2", " w3", " w4", " w5", " w6"]  # bounded history
    assert h["info"]["voice"] == "NATF2"
    assert lat["rows"][0]["latency_ms"] == "500"
    assert ev["type"] == "listening" and ev["session"] == "s1"
