"""End-to-end: real VoiceClient against the mock server, with fake audio devices."""

import asyncio
import csv
import io
import json

import numpy as np

from ava_voice import protocol
from ava_voice.config import DetectionConfig
from ava_voice.client import VoiceClient
from ava_voice.events import EventBus
from ava_voice.latency import LatencyTracker
from fakes import FakeAudio
from mock_server import MockServer


def run_conversation(tmp_path, **server_kw):
    async def go():
        srv = MockServer(**server_kw)
        port = await srv.start()
        out = io.StringIO()
        events = EventBus(stdout=out)
        audio = FakeAudio(script=((0.5, 0.3),))  # user talks for 0.5 s, then silence
        latency = LatencyTracker(tmp_path / "latency.csv")
        url = protocol.build_url("127.0.0.1", port, "NATF2", "You are AVA.")
        client = VoiceClient(url, audio, events, latency, DetectionConfig())
        stop = asyncio.Event()
        await asyncio.wait_for(client.run(stop, once=True), timeout=20)
        await srv.stop()
        return srv, audio, [json.loads(l) for l in out.getvalue().splitlines()]

    return asyncio.run(go())


def test_full_conversation(tmp_path):
    srv, audio, events = run_conversation(tmp_path)

    # The server got the persona via the query string.
    assert srv.queries == [{"voice_prompt": "NATF2.pt", "text_prompt": "You are AVA."}]
    # Mic audio reached the server (decoded from Ogg/Opus), and only after the handshake.
    assert srv.received_samples >= 24000
    assert srv.received_peak > 0.1
    assert not srv.audio_before_handshake

    types = [e["type"] for e in events]
    assert types[0] == "listening" and events[0]["reason"] == "session_start"
    assert types.count("robot_speaking_start") == 1
    assert types.count("robot_speaking_end") == 1
    i_start, i_end = types.index("robot_speaking_start"), types.index("robot_speaking_end")
    assert i_start < i_end
    assert events[i_end + 1]["type"] == "listening" and events[i_end + 1]["reason"] == "robot_finished"
    assert types[-1] == "session_closed" and events[-1]["reason"] == "server_closed"

    # Robot reply text: incremental deltas and the full turn text at the end.
    deltas = [e["delta"] for e in events if e["type"] == "robot_text"]
    assert deltas == ["Hello", " there", "."]
    assert all(e["turn"] == 1 for e in events if e["type"] == "robot_text")
    assert events[i_end]["text"] == "Hello there."
    # PersonaPlex gives no transcript of the user; no event may pretend otherwise.
    assert not any("user_text" in e or "transcript" in e for e in events)

    # All events of the session carry the same session id.
    assert len({e["session"] for e in events}) == 1 and events[0]["session"]

    # Robot audio was played.
    played = np.concatenate(audio.played)
    assert played.size > 24000 and np.abs(played).max() > 0.1
    assert audio.started and audio.stopped

    # Latency: one row, positive, and reported in the start event.
    rows = list(csv.DictReader((tmp_path / "latency.csv").open()))
    assert len(rows) == 1
    lat = int(rows[0]["latency_ms"])
    assert 0 < lat < 5000
    assert events[i_start]["latency_ms"] == lat


def test_connection_refused_is_handled(tmp_path):
    async def go():
        audio = FakeAudio()
        client = VoiceClient(protocol.build_url("127.0.0.1", 1, "NATF2", ""), audio, EventBus(stdout=None),
                             LatencyTracker(None), DetectionConfig())
        await asyncio.wait_for(client.run(asyncio.Event(), once=True), timeout=10)
        return audio

    audio = asyncio.run(go())
    assert audio.stopped


def test_stop_event_ends_session(tmp_path):
    async def go():
        srv = MockServer(close_after=False, user_audio_s=100)
        port = await srv.start()
        out = io.StringIO()
        client = VoiceClient(protocol.build_url("127.0.0.1", port, "VARM4", "x"), FakeAudio(),
                             EventBus(stdout=out), LatencyTracker(None), DetectionConfig())
        stop = asyncio.Event()
        task = asyncio.create_task(client.run(stop))
        await asyncio.sleep(1.0)
        stop.set()
        await asyncio.wait_for(task, timeout=5)
        await srv.stop()
        return [json.loads(l) for l in out.getvalue().splitlines()]

    events = asyncio.run(go())
    assert events[0]["type"] == "listening"
    assert events[-1] == {**events[-1], "type": "session_closed", "reason": "stopped"}
