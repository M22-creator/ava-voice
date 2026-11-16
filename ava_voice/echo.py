"""Echo handling: the robot's mic hears the robot's speaker.

ava-voice never mutes the mic while the robot talks: PersonaPlex is full
duplex and relies on hearing the user to be interrupted. Instead, echo must be
removed before the audio reaches us. Modes (config ``echo.mode``):

* ``none``: do nothing and print a warning.
* ``external``: echo is handled elsewhere (headset, echo-cancelling
  speakerphone, or a system-wide echo-cancel setup). Nothing to do here.
* ``pulse_echo_cancel``: load ``module-echo-cancel`` via ``pactl`` for the
  lifetime of the process (works with PulseAudio, and with PipeWire through
  pipewire-pulse), and point our streams at its virtual source/sink by
  setting ``PULSE_SOURCE``/``PULSE_SINK`` before PortAudio is loaded.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Callable, Optional

from .config import Config

log = logging.getLogger(__name__)

EC_SOURCE = "ava_ec_source"
EC_SINK = "ava_ec_sink"

NO_ECHO_WARNING = (
    "echo.mode is 'none': the robot's microphone will hear its own speaker. PersonaPlex may then "
    "react to its own voice (stop mid-sentence or talk to itself). Use a headset or an "
    "echo-cancelling speakerphone (echo.mode: external) or echo.mode: pulse_echo_cancel. "
    "See README, section 'Echo'."
)

Runner = Callable[[list[str]], str]


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pulse_module_args(cfg: Config) -> list[str]:
    p = cfg.echo.pulse
    args = [
        "module-echo-cancel",
        f"aec_method={p.aec_method}",
        f"source_name={EC_SOURCE}",
        f"sink_name={EC_SINK}",
    ]
    if p.source_master:
        args.append(f"source_master={p.source_master}")
    if p.sink_master:
        args.append(f"sink_master={p.sink_master}")
    if p.aec_args:
        args.append(f"aec_args={p.aec_args}")
    return args


class EchoSetup:
    def __init__(self, cfg: Config, runner: Runner = _run, environ=os.environ):
        self.cfg = cfg
        self.runner = runner
        self.environ = environ
        self.module_index: Optional[str] = None

    def apply(self) -> None:
        mode = self.cfg.echo.mode
        if mode == "none":
            log.warning(NO_ECHO_WARNING)
        elif mode == "external":
            log.info("echo.mode is 'external': assuming the audio devices cancel echo themselves")
        elif mode == "pulse_echo_cancel":
            self._load_pulse()

    def _load_pulse(self) -> None:
        if self.runner is _run and shutil.which("pactl") is None:
            raise RuntimeError(
                "echo.mode is 'pulse_echo_cancel' but 'pactl' was not found. Install pulseaudio-utils "
                "(works with PipeWire too), or use another echo.mode."
            )
        try:
            out = self.runner(["pactl", "load-module", *pulse_module_args(self.cfg)])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"pactl could not load module-echo-cancel: {(e.stderr or '').strip() or e}. "
                "Check source_master/sink_master names with `pactl list short sources` / `sinks`."
            ) from e
        self.module_index = out.strip()
        # Read by the PulseAudio client library (and pipewire-pulse) when our streams open.
        self.environ["PULSE_SOURCE"] = EC_SOURCE
        self.environ["PULSE_SINK"] = EC_SINK
        a = self.cfg.audio
        if a.input_device is None:
            a.input_device = "pulse"
        if a.output_device is None:
            a.output_device = "pulse"
        log.info("loaded module-echo-cancel (index %s); using %s / %s", self.module_index, EC_SOURCE, EC_SINK)

    def undo(self) -> None:
        if self.module_index:
            try:
                self.runner(["pactl", "unload-module", self.module_index])
            except (subprocess.CalledProcessError, OSError) as e:
                log.warning("could not unload module-echo-cancel %s: %s", self.module_index, e)
            self.module_index = None
