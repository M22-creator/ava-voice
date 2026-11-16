"""Start and stop the upstream PersonaPlex server (``python -m moshi.server``)."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import ServerConfig

log = logging.getLogger(__name__)


def build_server_argv(cfg: ServerConfig, python: str = sys.executable) -> list[str]:
    # --static none: no web UI (we are headless), so upstream does not download dist.tgz.
    # No --ssl: the client is on the same machine, and upstream's --ssl downloads
    #   mkcert and installs a local root CA, which we do not want on the robot.
    # --host only changes what upstream prints; aiohttp is started without a host,
    #   so the server listens on all interfaces either way (see README, Security).
    argv = [python, "-m", "moshi.server", "--host", cfg.host, "--port", str(cfg.port), "--static", "none"]
    if cfg.cpu_offload:
        argv.append("--cpu-offload")
    argv += [str(a) for a in cfg.extra_args]
    return argv


class ServerError(RuntimeError):
    pass


class ServerProcess:
    def __init__(self, argv: list[str], log_path: Path):
        self.argv = argv
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("ab")
        log.info("starting PersonaPlex server: %s (log: %s)", " ".join(self.argv), self.log_path)
        # stdout of the server goes to the log file so our stdout carries only events.
        self.proc = subprocess.Popen(self.argv, stdout=self._log_fh, stderr=subprocess.STDOUT)

    async def wait_ready(self, host: str, port: int, timeout_s: float) -> None:
        """Wait until the server accepts TCP connections (it binds only after model load + warmup)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        next_note = loop.time() + 30
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(
                    f"PersonaPlex server exited with code {self.proc.returncode} before it was ready.\n"
                    f"Last lines of {self.log_path}:\n{self.tail()}"
                )
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2)
                writer.close()
                return
            except (OSError, asyncio.TimeoutError):
                pass
            now = loop.time()
            if now > deadline:
                raise ServerError(f"PersonaPlex server not ready after {timeout_s:.0f}s; see {self.log_path}")
            if now > next_note:
                log.info("still waiting for the server (loading/downloading the model)...")
                next_note = now + 30
            await asyncio.sleep(1.0)

    def tail(self, n: int = 20) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except FileNotFoundError:
            return "(no log)"
        return "\n".join(lines[-n:])

    def stop(self, timeout_s: float = 10.0) -> None:
        if self.proc is not None and self.proc.poll() is None:
            log.info("stopping PersonaPlex server")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
