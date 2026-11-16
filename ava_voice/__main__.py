"""Entry point: ``python -m ava_voice`` (what run.sh calls)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from .config import ConfigError, load_config

log = logging.getLogger("ava_voice")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ava_voice", description="Talk to AVA through PersonaPlex-7B.")
    p.add_argument("--config", type=Path, default=None, help="config file (default: config.yaml in the repo)")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--no-server", action="store_true", help="do not start moshi.server; connect to a running one")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


async def amain(cfg, launch_server: bool) -> int:
    # Imported here so --list-devices and config errors do not need aiohttp/sphn.
    from . import protocol
    from .audio import SoundDeviceIO
    from .client import VoiceClient
    from .events import EventBus
    from .latency import LatencyTracker
    from .server_proc import ServerError, ServerProcess, build_server_argv

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    events = EventBus.from_config(cfg.events)
    await events.start_socket(cfg.events.socket)
    server = None
    try:
        if launch_server:
            server = ServerProcess(build_server_argv(cfg.server), cfg.resolve(cfg.server.log_file))
            server.start()
            ready = asyncio.create_task(server.wait_ready(cfg.server.host, cfg.server.port, cfg.server.startup_timeout_s))
            stopped = asyncio.create_task(stop.wait())
            await asyncio.wait({ready, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                ready.cancel()
                return 130
            stopped.cancel()
            ready.result()  # re-raises ServerError
            log.info("PersonaPlex server is up on port %d", cfg.server.port)

        url = protocol.build_url(cfg.server.host, cfg.server.port, cfg.persona.voice, cfg.role_prompt())
        audio = SoundDeviceIO(cfg.audio.input_device, cfg.audio.output_device, cfg.audio.playback_max_buffer_s)
        latency = LatencyTracker(cfg.resolve(cfg.latency.csv_path) if cfg.latency.csv_path else None)
        client = VoiceClient(url, audio, events, latency, cfg.detection)
        log.info("voice=%s, role prompt from %s", cfg.persona.voice, cfg.persona.role_prompt_file)
        await client.run(stop)
        return 0
    except ServerError as e:
        log.error("%s", e)
        return 1
    finally:
        if server is not None:
            server.stop()
        await events.close()


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,  # stdout is reserved for JSON events
    )
    try:
        cfg = load_config(args.config)
        cfg.role_prompt()  # fail early if the prompt file is missing
    except ConfigError as e:
        log.error("config error: %s", e)
        return 2

    from .echo import EchoSetup

    # Echo setup may set PULSE_SOURCE/PULSE_SINK, which must happen before PortAudio loads.
    echo = EchoSetup(cfg)
    try:
        if args.list_devices:
            from .audio import list_devices

            print(list_devices())
            return 0
        echo.apply()
        return asyncio.run(amain(cfg, launch_server=cfg.server.launch and not args.no_server))
    except RuntimeError as e:
        log.error("%s", e)
        return 1
    finally:
        echo.undo()


if __name__ == "__main__":
    sys.exit(main())
