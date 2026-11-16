"""Headless PersonaPlex client: mic -> server -> speaker, plus events and latency."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import aiohttp
import numpy as np
import sphn

from . import protocol
from .audio import AudioIO
from .config import DetectionConfig
from .detect import SpeechActivity
from .events import EventBus
from .latency import LatencyTracker
from .protocol import FRAME_SIZE, SAMPLE_RATE

log = logging.getLogger(__name__)

MIC_QUEUE_MAX = 50  # 4 s of 80 ms chunks; older chunks are dropped if sending stalls


class HandshakeError(Exception):
    pass


class VoiceClient:
    def __init__(
        self,
        url: str,
        audio: AudioIO,
        events: EventBus,
        latency: LatencyTracker,
        detection: DetectionConfig,
        handshake_timeout_s: float = 120.0,
    ):
        self.url = url
        self.audio = audio
        self.events = events
        self.latency = latency
        self.detection = detection
        self.handshake_timeout_s = handshake_timeout_s
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mic_q: Optional[asyncio.Queue] = None
        self._ready = False
        self.turn = 0
        self._reset_session_state()

    def _reset_session_state(self) -> None:
        d = self.detection
        self.robot_det = SpeechActivity(d.robot_speech_threshold_dbfs, d.robot_speech_hangover_s)
        self.user_det = SpeechActivity(d.user_speech_threshold_dbfs, d.user_speech_hangover_s)
        self._turn_text: list[str] = []

    # ---- mic side (called from the audio thread) ----------------------------

    def _on_mic(self, pcm: np.ndarray, t0: float) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._enqueue_mic, pcm, t0)

    def _enqueue_mic(self, pcm: np.ndarray, t0: float) -> None:
        if not self._ready or self._mic_q is None:
            return  # no session yet: the server must not receive audio before the handshake
        if self._mic_q.full():
            self._mic_q.get_nowait()
        self._mic_q.put_nowait((pcm, t0))

    # ---- public entry points -------------------------------------------------

    async def run(self, stop: asyncio.Event, once: bool = False, retry_s: float = 2.0) -> None:
        """Run conversations until ``stop`` is set, reconnecting if the server drops."""
        self._loop = asyncio.get_running_loop()
        self._mic_q = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
        self.audio.start(self._on_mic)
        try:
            async with aiohttp.ClientSession() as http:
                while not stop.is_set():
                    try:
                        await self._session(http, stop)
                    except (aiohttp.ClientError, HandshakeError, asyncio.TimeoutError, OSError) as e:
                        log.warning("connection to %s failed: %s", self.url.split("?")[0], e)
                    if once or stop.is_set():
                        break
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=retry_s)
                    except asyncio.TimeoutError:
                        pass
        finally:
            self.audio.stop()

    # ---- one conversation ----------------------------------------------------

    async def _session(self, http: aiohttp.ClientSession, stop: asyncio.Event) -> None:
        log.info("connecting to %s", self.url.split("?")[0])
        async with http.ws_connect(self.url, max_msg_size=0) as ws:
            log.info("connected; waiting for the server to process the voice and role prompts")
            await self._wait_handshake(ws)
            self.events.session = uuid.uuid4().hex[:12]
            self._reset_session_state()
            opus_writer = sphn.OpusStreamWriter(SAMPLE_RATE)
            opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
            while not self._mic_q.empty():
                self._mic_q.get_nowait()
            self._ready = True
            log.info("session %s started", self.events.session)
            self.events.emit("listening", reason="session_start")

            tasks = {
                asyncio.create_task(self._send_loop(ws, opus_writer), name="send"),
                asyncio.create_task(self._recv_loop(ws, opus_reader), name="recv"),
                asyncio.create_task(stop.wait(), name="stop"),
            }
            reason = "stopped"
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                finished = next(iter(done))
                if finished.get_name() == "recv":
                    reason = "server_closed"
                elif finished.get_name() == "send":
                    reason = "error"
                if not finished.cancelled() and finished.exception() is not None:
                    reason = "error"
                    log.error("session error: %r", finished.exception())
            finally:
                self._ready = False
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._end_session(reason)

    async def _wait_handshake(self, ws) -> None:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=self.handshake_timeout_s)
        except asyncio.TimeoutError:
            raise HandshakeError(
                f"no handshake within {self.handshake_timeout_s:.0f}s (is another client connected? "
                "the server serves one conversation at a time)"
            ) from None
        if msg.type != aiohttp.WSMsgType.BINARY:
            raise HandshakeError(f"expected a binary handshake, got {msg.type!r} (check logs/server.log)")
        kind, _ = protocol.decode(msg.data)
        if kind != protocol.MSG_HANDSHAKE:
            raise HandshakeError(f"expected handshake (0x00), got message kind {kind}")

    async def _send_loop(self, ws, opus_writer) -> None:
        # sphn's Opus writer only accepts Opus frame sizes, so re-chunk whatever
        # the audio device delivers into 80 ms frames (1920 samples, one model frame).
        pending = np.zeros(0, dtype=np.float32)
        pending_t0 = 0.0
        while True:
            pcm, t0 = await self._mic_q.get()
            if pending.size == 0:
                pending_t0 = t0
            pending = np.concatenate([pending, np.asarray(pcm, dtype=np.float32).reshape(-1)])
            while pending.size >= FRAME_SIZE:
                frame, pending = pending[:FRAME_SIZE], pending[FRAME_SIZE:]
                frame_t0, pending_t0 = pending_t0, pending_t0 + FRAME_SIZE / SAMPLE_RATE
                for kind, t in self.user_det.update(frame, frame_t0, SAMPLE_RATE):
                    if kind == "start":
                        self.latency.on_user_start(t)
                    else:
                        self.latency.on_user_end(t)
                opus_writer.append_pcm(frame)
            data = opus_writer.read_bytes()
            if data:
                await ws.send_bytes(protocol.encode_audio(data))

    async def _recv_loop(self, ws, opus_reader) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                if not msg.data:
                    continue
                kind, payload = protocol.decode(msg.data)
                if kind == protocol.MSG_AUDIO:
                    opus_reader.append_bytes(payload)
                    pcm = opus_reader.read_pcm()
                    if pcm.size:
                        self._on_robot_audio(pcm)
                elif kind == protocol.MSG_TEXT:
                    self._on_robot_text(payload.decode("utf-8", errors="replace"))
                elif kind == protocol.MSG_HANDSHAKE:
                    pass
                else:
                    log.debug("ignoring message kind %d", kind)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                log.error("websocket error: %s", ws.exception())
                return

    # ---- robot side ----------------------------------------------------------

    def _on_robot_audio(self, pcm: np.ndarray) -> None:
        # When this chunk will actually come out of the speaker.
        t_audible = time.time() + self.audio.buffered_seconds() + self.audio.output_latency()
        self.audio.play(pcm)
        for kind, t in self.robot_det.update(pcm, t_audible, SAMPLE_RATE):
            if kind == "start":
                self._robot_start(t)
            else:
                self._robot_end(t)

    def _robot_start(self, t_audible: float) -> None:
        self.turn += 1
        row = self.latency.on_robot_start(t_audible, self.events.session, self.turn)
        self.events.emit(
            "robot_speaking_start",
            turn=self.turn,
            audible_at=round(t_audible, 3),
            latency_ms=row["latency_ms"] if row else None,
        )

    def _robot_end(self, t_audible: float, session_ending: bool = False) -> None:
        text = "".join(self._turn_text).strip()
        self._turn_text = []
        self.latency.on_robot_end(user_still_active=self.user_det.active)
        self.events.emit("robot_speaking_end", turn=self.turn, audible_until=round(t_audible, 3), text=text)
        if not session_ending:
            self.events.emit("listening", reason="robot_finished")

    def _on_robot_text(self, piece: str) -> None:
        self._turn_text.append(piece)
        turn = self.turn if self.robot_det.active else self.turn + 1
        self.events.emit("robot_text", turn=turn, delta=piece)

    def _end_session(self, reason: str) -> None:
        for _, t in self.robot_det.force_end():
            self._robot_end(t, session_ending=True)
        self.audio.clear_playback()
        self.events.emit("session_closed", reason=reason)
        log.info("session %s closed (%s)", self.events.session, reason)
