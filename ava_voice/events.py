"""JSON-lines events for the robot state machine.

Each event is one JSON object on one line, written to stdout and/or to every
client connected to an optional local socket. Schema: README.md, "Events".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Optional, TextIO

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class EventBus:
    def __init__(self, stdout: Optional[TextIO] = None, session: Optional[str] = None):
        self._stdout = stdout
        self._writers: set[asyncio.StreamWriter] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self._unix_path: Optional[str] = None
        self._listeners: list[Callable[[dict], None]] = []
        self.session = session

    @classmethod
    def from_config(cls, events_cfg) -> "EventBus":
        return cls(stdout=sys.stdout if events_cfg.stdout else None)

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def emit(self, type_: str, **fields: Any) -> dict:
        event = {"v": SCHEMA_VERSION, "type": type_, "ts": round(time.time(), 3), "session": self.session}
        event.update(fields)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        if self._stdout is not None:
            self._stdout.write(line)
            self._stdout.flush()
        data = line.encode("utf-8")
        for w in list(self._writers):
            if w.is_closing():
                self._writers.discard(w)
                continue
            try:
                w.write(data)
            except (ConnectionError, RuntimeError):
                self._writers.discard(w)
        for fn in self._listeners:
            fn(event)
        return event

    async def start_socket(self, spec: Optional[str]) -> None:
        """Start the optional socket. ``spec`` is 'unix:/path' or 'tcp:HOST:PORT'.
        Connected clients only receive; anything they send is ignored."""
        if not spec:
            return

        async def on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            self._writers.add(writer)
            try:
                while await reader.read(1024):
                    pass
            except ConnectionError:
                pass
            finally:
                self._writers.discard(writer)
                writer.close()

        if spec.startswith("unix:"):
            path = spec[len("unix:"):]
            if os.path.exists(path):
                os.unlink(path)  # stale socket from a previous run
            self._server = await asyncio.start_unix_server(on_client, path=path)
            self._unix_path = path
        elif spec.startswith("tcp:"):
            host, _, port = spec[len("tcp:"):].rpartition(":")
            self._server = await asyncio.start_server(on_client, host=host or "127.0.0.1", port=int(port))
        else:
            raise ValueError(f"bad events socket spec: {spec}")
        log.info("event socket listening on %s", spec)

    async def close(self) -> None:
        for w in list(self._writers):
            w.close()
        self._writers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._unix_path and os.path.exists(self._unix_path):
            os.unlink(self._unix_path)
