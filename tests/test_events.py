import asyncio
import io
import json
import os
import tempfile

from ava_voice.events import EventBus


def test_stdout_json_lines():
    out = io.StringIO()
    bus = EventBus(stdout=out, session="abc")
    bus.emit("listening", reason="session_start")
    bus.emit("robot_text", turn=1, delta=" héllo")
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    e0, e1 = map(json.loads, lines)
    assert e0["type"] == "listening" and e0["session"] == "abc" and e0["v"] == 1
    assert isinstance(e0["ts"], float)
    assert e1["delta"] == " héllo"
    assert "user_text" not in e1 and "transcript" not in e1


def test_unix_socket_receives_events():
    # AF_UNIX paths are limited to ~104 bytes; pytest's tmp_path can be longer on macOS.
    short_dir = tempfile.mkdtemp(dir="/tmp" if os.path.isdir("/tmp") else None)

    async def run():
        path = os.path.join(short_dir, "ev.sock")
        bus = EventBus(stdout=None)
        await bus.start_socket(f"unix:{path}")
        reader, writer = await asyncio.open_unix_connection(path)
        await asyncio.sleep(0.05)
        bus.emit("robot_speaking_start", turn=1)
        line = await asyncio.wait_for(reader.readline(), 2)
        writer.close()
        await bus.close()
        assert not os.path.exists(path)  # cleaned up on close
        return json.loads(line)

    ev = asyncio.run(run())
    os.rmdir(short_dir)
    assert ev["type"] == "robot_speaking_start" and ev["turn"] == 1


def test_tcp_socket_and_disconnected_clients_are_dropped():
    async def run():
        bus = EventBus(stdout=None)
        await bus.start_socket("tcp:127.0.0.1:0")
        port = bus._server.sockets[0].getsockname()[1]
        r1, w1 = await asyncio.open_connection("127.0.0.1", port)
        r2, w2 = await asyncio.open_connection("127.0.0.1", port)
        await asyncio.sleep(0.05)
        w1.close()
        await asyncio.sleep(0.05)
        bus.emit("listening")
        bus.emit("listening")
        line = await asyncio.wait_for(r2.readline(), 2)
        w2.close()
        await bus.close()
        return json.loads(line)

    assert asyncio.run(run())["type"] == "listening"
