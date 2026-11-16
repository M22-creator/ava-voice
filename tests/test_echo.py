import subprocess
from pathlib import Path

import pytest

from ava_voice.config import load_config
from ava_voice.echo import EC_SINK, EC_SOURCE, EchoSetup, pulse_module_args


def cfg_with(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return load_config(p)


def test_none_mode_warns_and_does_nothing(tmp_path, caplog):
    calls = []
    env = {}
    EchoSetup(cfg_with(tmp_path, "echo: {mode: none}\n"), runner=calls.append, environ=env).apply()
    assert calls == [] and env == {}
    assert "will hear its own speaker" in caplog.text


def test_pulse_mode_loads_and_unloads_module(tmp_path):
    cfg = cfg_with(tmp_path, "echo:\n  mode: pulse_echo_cancel\n  pulse:\n    source_master: mic0\n    sink_master: spk0\n")
    calls = []

    def runner(cmd):
        calls.append(cmd)
        return "42\n"

    env = {}
    e = EchoSetup(cfg, runner=runner, environ=env)
    e.apply()
    assert calls[0][:2] == ["pactl", "load-module"]
    assert "module-echo-cancel" in calls[0]
    assert "source_master=mic0" in calls[0] and "sink_master=spk0" in calls[0]
    assert env == {"PULSE_SOURCE": EC_SOURCE, "PULSE_SINK": EC_SINK}
    assert cfg.audio.input_device == "pulse" and cfg.audio.output_device == "pulse"
    e.undo()
    assert calls[1] == ["pactl", "unload-module", "42"]
    e.undo()  # idempotent
    assert len(calls) == 2


def test_pulse_mode_keeps_explicit_devices(tmp_path):
    cfg = cfg_with(tmp_path, "echo: {mode: pulse_echo_cancel}\naudio: {input_device: pipewire, output_device: 3}\n")
    EchoSetup(cfg, runner=lambda c: "1", environ={}).apply()
    assert cfg.audio.input_device == "pipewire" and cfg.audio.output_device == 3
    assert not any(a.startswith("source_master") for a in pulse_module_args(cfg))


def test_pulse_failure_is_explained(tmp_path):
    cfg = cfg_with(tmp_path, "echo: {mode: pulse_echo_cancel}\n")

    def runner(cmd):
        raise subprocess.CalledProcessError(1, cmd, stderr="Failure: No such entity")

    with pytest.raises(RuntimeError, match="No such entity"):
        EchoSetup(cfg, runner=runner, environ={}).apply()


def test_client_never_mutes_mic():
    # Requirement: do not silently mute the mic while the robot talks.
    # The send loop must forward every mic chunk regardless of robot state.
    src = (Path(__file__).parent.parent / "ava_voice/client.py").read_text()
    send_loop = src.split("async def _send_loop", 1)[1].split("async def _recv_loop", 1)[0]
    assert "robot_det" not in send_loop and "robot_speaking" not in send_loop
