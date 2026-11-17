from ava_voice.__main__ import main


def test_bad_config_exits_2(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("persona: {voice: NOPE}\n")
    assert main(["--config", str(p)]) == 2


def test_missing_prompt_exits_2(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("persona: {role_prompt_file: missing.txt}\n")
    assert main(["--config", str(p)]) == 2


def test_launch_server_run_client_and_shut_down(tmp_path, monkeypatch, capsys):
    """amain: start a server subprocess (the mock), wait for its port, converse, stop on SIGTERM."""
    import asyncio
    import json
    import os
    import signal
    import socket
    import sys
    from pathlib import Path

    import ava_voice.audio
    import ava_voice.server_proc
    from ava_voice.__main__ import amain
    from ava_voice.config import load_config
    from fakes import FakeAudio

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    (tmp_path / "p.txt").write_text("You are AVA.\n")
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text(
        f"server: {{port: {port}, startup_timeout_s: 30, log_file: srv.log}}\n"
        "persona: {role_prompt_file: p.txt}\nlatency: {csv_path: lat.csv}\nweb: {port: 0}\n"
    )
    cfg = load_config(cfgp)
    mock = str(Path(__file__).parent / "mock_server.py")
    monkeypatch.setattr(ava_voice.server_proc, "build_server_argv", lambda c: [sys.executable, mock, str(c.port)])
    monkeypatch.setattr(ava_voice.audio, "SoundDeviceIO", lambda *a: FakeAudio(script=((0.5, 0.3),)))

    async def go():
        asyncio.get_running_loop().call_later(6.0, os.kill, os.getpid(), signal.SIGTERM)
        return await amain(cfg, launch_server=True)

    assert asyncio.run(go()) == 0
    events = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    types = [e["type"] for e in events]
    assert "robot_speaking_start" in types and "robot_speaking_end" in types
    assert types[-1] == "session_closed"
    assert (tmp_path / "lat.csv").exists()
    assert "mock server on" in (tmp_path / "srv.log").read_text()
