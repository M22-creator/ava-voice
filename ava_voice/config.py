"""Load and validate config.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .protocol import VOICES

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

ECHO_MODES = ("none", "external", "pulse_echo_cancel")

DeviceSpec = Optional[Union[int, str]]


class ConfigError(Exception):
    pass


@dataclass
class ServerConfig:
    launch: bool = True
    host: str = "127.0.0.1"
    port: int = 8998
    cpu_offload: bool = False
    startup_timeout_s: float = 900.0
    log_file: str = "logs/server.log"
    extra_args: list = field(default_factory=list)


@dataclass
class PersonaConfig:
    voice: str = "NATF2"
    role_prompt_file: str = "prompts/ava_role.txt"


@dataclass
class AudioConfig:
    input_device: DeviceSpec = None
    output_device: DeviceSpec = None
    # Largest amount of robot audio we queue for the speaker. Anything beyond
    # this is dropped (oldest first) so latency cannot creep up.
    playback_max_buffer_s: float = 1.0


@dataclass
class PulseEchoCancelConfig:
    source_master: Optional[str] = None
    sink_master: Optional[str] = None
    aec_method: str = "webrtc"
    aec_args: str = ""


@dataclass
class EchoConfig:
    mode: str = "none"
    pulse: PulseEchoCancelConfig = field(default_factory=PulseEchoCancelConfig)


@dataclass
class EventsConfig:
    stdout: bool = True
    socket: Optional[str] = None  # "unix:/path" or "tcp:HOST:PORT"


@dataclass
class DetectionConfig:
    robot_speech_threshold_dbfs: float = -40.0
    robot_speech_hangover_s: float = 0.5
    user_speech_threshold_dbfs: float = -40.0
    user_speech_hangover_s: float = 0.4


@dataclass
class WebConfig:
    enabled: bool = True
    host: str = "127.0.0.1"   # local only: the page has no login
    port: int = 8002
    history: int = 300        # events kept for a page that connects later


@dataclass
class AsrConfig:
    # Optional transcript of what the USER said, from a separate speech recognizer
    # (PersonaPlex itself produces none). Emits user_text events.
    enabled: bool = False
    engine: str = "parakeet"
    model_dir: str = "models"
    threads: int = 2          # kept low: shares the CPU with the robot's other software
    min_speech_s: float = 0.3  # ignore shorter blips
    max_speech_s: float = 20.0


@dataclass
class LatencyConfig:
    csv_path: Optional[str] = "logs/latency.csv"


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    echo: EchoConfig = field(default_factory=EchoConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    web: WebConfig = field(default_factory=WebConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    base_dir: Path = REPO_ROOT

    def resolve(self, path: str) -> Path:
        """Resolve a path from the config relative to the config file's directory."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else self.base_dir / p

    def role_prompt(self) -> str:
        return read_prompt(self.resolve(self.persona.role_prompt_file))


def read_prompt(path: Path) -> str:
    """Read a prompt file. Lines starting with '#' are comments and are dropped;
    the remaining lines are joined into one paragraph, which is the form the
    upstream examples use."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"role prompt file not found: {path}") from None
    lines = [ln.strip() for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return " ".join(ln for ln in lines if ln)


def _build(cls, data: Any, where: str):
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"'{where}' must be a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in fields:
            raise ConfigError(f"unknown config key '{where}.{key}'")
        f = fields[key]
        factory = f.default_factory
        if isinstance(factory, type) and dataclasses.is_dataclass(factory):
            value = _build(factory, value, f"{where}.{key}")
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Optional[Path] = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse {path}: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    sections = {
        "server": ServerConfig,
        "persona": PersonaConfig,
        "audio": AudioConfig,
        "echo": EchoConfig,
        "events": EventsConfig,
        "detection": DetectionConfig,
        "latency": LatencyConfig,
        "web": WebConfig,
        "asr": AsrConfig,
    }
    kwargs = {}
    for key, value in raw.items():
        if key not in sections:
            raise ConfigError(f"unknown config section '{key}'")
        kwargs[key] = _build(sections[key], value, key)
    cfg = Config(**kwargs, base_dir=path.resolve().parent)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    if cfg.persona.voice not in VOICES:
        raise ConfigError(
            f"persona.voice '{cfg.persona.voice}' is not a PersonaPlex voice. "
            f"Choose one of: {', '.join(VOICES)}"
        )
    if cfg.echo.mode not in ECHO_MODES:
        raise ConfigError(f"echo.mode must be one of {ECHO_MODES}, got '{cfg.echo.mode}'")
    sock = cfg.events.socket
    if sock is not None and not (sock.startswith("unix:") or sock.startswith("tcp:")):
        raise ConfigError("events.socket must look like 'unix:/path/to.sock' or 'tcp:127.0.0.1:8765'")
    for name in ("input_device", "output_device"):
        v = getattr(cfg.audio, name)
        if v is not None and not isinstance(v, (int, str)):
            raise ConfigError(f"audio.{name} must be a device index, a name, or null")
    if not isinstance(cfg.server.extra_args, list):
        raise ConfigError("server.extra_args must be a list of strings")
    if cfg.asr.engine not in ("parakeet",):
        raise ConfigError(f"asr.engine must be 'parakeet', got '{cfg.asr.engine}'")
    if not (0 <= cfg.web.port < 65536):
        raise ConfigError("web.port must be 0-65535 (0 = any free port)")
    if cfg.audio.playback_max_buffer_s <= 0:
        raise ConfigError("audio.playback_max_buffer_s must be > 0")
