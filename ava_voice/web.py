"""Live conversation page for ava-voice:  http://127.0.0.1:8002 (config `web:`).

Runs inside the client's event loop (aiohttp, already a dependency). The page
shows the conversation state, the robot's reply text, the user's words if the
ASR sidecar is enabled, and response latency per turn.

    GET /             the page
    GET /events       Server-Sent Events: every event, as JSON, as it happens
    GET /api/history  recent events (so a page opened later can catch up)
    GET /api/latency  rows of the latency CSV, newest last

Local only by default (127.0.0.1): it has no login. From a laptop:
ssh -L 8002:localhost:8002 <user>@<robot>.
"""
from __future__ import annotations

import asyncio
import collections
import csv
import json
import logging
from pathlib import Path
from typing import Deque, Optional, Set

from aiohttp import web

from .events import EventBus

log = logging.getLogger(__name__)
PAGE = Path(__file__).resolve().parent / "web" / "index.html"


class WebView:
    def __init__(self, events: EventBus, latency_csv: Optional[Path], host: str = "127.0.0.1",
                 port: int = 8002, history: int = 300, info: Optional[dict] = None) -> None:
        self.latency_csv = latency_csv
        self.host, self.port = host, port
        self.info = info or {}
        self.history: Deque[dict] = collections.deque(maxlen=history)
        self.clients: Set[asyncio.Queue] = set()
        self.runner: Optional[web.AppRunner] = None
        events.subscribe(self._on_event)

    def _on_event(self, event: dict) -> None:
        self.history.append(event)
        for q in list(self.clients):
            if q.full():  # a stalled browser tab must not grow memory without bound
                self.clients.discard(q)
                continue
            q.put_nowait(event)

    # ---- handlers ----------------------------------------------------------

    async def page(self, request: web.Request) -> web.Response:
        return web.Response(body=PAGE.read_bytes(), content_type="text/html", charset="utf-8",
                            headers={"Cache-Control": "no-store"})

    async def api_history(self, request: web.Request) -> web.Response:
        return web.json_response({"events": list(self.history), "info": self.info})

    async def api_latency(self, request: web.Request) -> web.Response:
        rows = []
        if self.latency_csv and self.latency_csv.exists():
            with self.latency_csv.open(newline="") as f:
                rows = list(csv.DictReader(f))[-500:]
        return web.json_response({"rows": rows})

    async def sse(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-store"})
        await resp.prepare(request)
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.clients.add(q)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15)
                    await resp.write(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n")
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(q)
        return resp

    # ---- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.page)
        app.router.add_get("/events", self.sse)
        app.router.add_get("/api/history", self.api_history)
        app.router.add_get("/api/latency", self.api_latency)
        self.runner = web.AppRunner(app, handle_signals=False)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        log.info("live page: http://%s:%d", self.host, self.port)
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            log.warning("web.host is %s: the page has no login; anyone who can reach it sees the conversation", self.host)

    async def close(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
