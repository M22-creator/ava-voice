from pathlib import Path

import pytest

from ava_voice.config import ConfigError, load_config, read_prompt

REPO = Path(__file__).parent.parent


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_repo_config_loads_and_prompt_is_clean():
    cfg = load_config(REPO / "config.yaml")
    assert cfg.persona.voice == "NATF2"
    assert cfg.echo.mode == "none"
    prompt = cfg.role_prompt()
    assert "AVA" in prompt
    assert "#" not in prompt and "DRAFT" not in prompt  # comments are not sent to the model
    assert "\n" not in prompt
    # Rough budget check: well under the 512 pinned frames upstream allows by default
    # (~1 frame per token; English averages < 1.5 tokens per word).
    assert len(prompt.split()) < 200


def test_defaults_for_empty_file(tmp_path):
    cfg = load_config(write(tmp_path, ""))
    assert cfg.server.port == 8998 and cfg.server.launch is True
    assert cfg.audio.input_device is None


def test_relative_paths_resolve_to_config_dir(tmp_path):
    cfg = load_config(write(tmp_path, "persona:\n  role_prompt_file: p.txt\n"))
    (tmp_path / "p.txt").write_text("# comment\nHello\n\n  world  \n")
    assert cfg.role_prompt() == "Hello world"


@pytest.mark.parametrize("text, msg", [
    ("persona:\n  voice: BOB\n", "not a PersonaPlex voice"),
    ("echo:\n  mode: mute\n", "echo.mode"),
    ("serverr: {}\n", "unknown config section"),
    ("server:\n  prot: 1\n", "unknown config key 'server.prot'"),
    ("echo:\n  pulse:\n    bogus: 1\n", "echo.pulse.bogus"),
    ("events:\n  socket: /tmp/x.sock\n", "events.socket"),
    ("- a\n- b\n", "mapping"),
])
def test_validation_errors(tmp_path, text, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(write(tmp_path, text))


def test_nested_pulse_section(tmp_path):
    cfg = load_config(write(tmp_path, "echo:\n  mode: pulse_echo_cancel\n  pulse:\n    source_master: alsa_input.usb\n"))
    assert cfg.echo.pulse.source_master == "alsa_input.usb"
    assert cfg.echo.pulse.aec_method == "webrtc"


def test_missing_prompt_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        read_prompt(tmp_path / "nope.txt")
