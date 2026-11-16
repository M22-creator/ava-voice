"""A stand-in for upstream ``moshi.server`` that needs no GPU.

It follows the behaviour of ``ServerState.handle_chat`` at the pinned commit:
reads ``voice_prompt`` and ``text_prompt`` from the query (KeyError -> HTTP 500
if missing, like upstream), sends ``0x00`` after a "prompt processing" delay,
decodes ``0x01`` Ogg/Opus audio from the client, and streams ``0x01`` audio and
``0x02`` text back. Instead of a model, it plays a scripted reply: after the
client has sent ``user_audio_s`` seconds of audio, it sends a tone (the
"robot speaking") with some text pieces, then silence, then closes.
"""

from __future__ import annotations

import asyncio
import sys

import numpy as np
import sphn
from aiohttp import web

SR = 24000
FRAME = 1920


def tone(n: int, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class MockServer:
    def __init__(self, handshake_delay_s=0.3, user_audio_s=1.0, reply_s=0.6, tail_s=1.0,
                 text=("Hello", " there", "."), frame_period_s=FRAME / SR, close_after=True):
        self.handshake_delay_s = handshake_delay_s
        self.user_audio_s = user_audio_s
        self.reply_s = reply_s
        self.tail_s = tail_s
        self.text = text
        self.frame_period_s = frame_period_s
        self.close_after = close_after
        self.queries: list[dict] = []
        self.received_samples = 0
        self.received_peak = 0.0
        self.audio_before_handshake = False
        self.runner: web.AppRunner | None = None
        self.port: int | None = None

    async def handle_chat(self, request: web.Request):
        voice = request.query["voice_prompt"]  # KeyError -> 500, as upstream
        text_prompt = request.query["text_prompt"]
        self.queries.append({"voice_prompt": voice, "text_prompt": text_prompt})
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        reader = sphn.OpusStreamReader(SR)
        writer = sphn.OpusStreamWriter(SR)
        handshake_sent = False

        async def recv():
            async for msg in ws:
                data = msg.data
                if isinstance(data, (bytes, bytearray)) and data and data[0] == 1:
                    if not handshake_sent:
                        self.audio_before_handshake = True
                    reader.append_bytes(data[1:])
                    pcm = reader.read_pcm()
                    if pcm.size:
                        self.received_samples += pcm.size
                        self.received_peak = max(self.received_peak, float(np.abs(pcm).max()))

        recv_task = asyncio.create_task(recv())
        await asyncio.sleep(self.handshake_delay_s)
        await ws.send_bytes(b"\x00")
        handshake_sent = True

        async def send_frame(pcm):
            writer.append_pcm(pcm)
            data = writer.read_bytes()
            if data:
                await ws.send_bytes(b"\x01" + data)
            await asyncio.sleep(self.frame_period_s)

        silence = np.zeros(FRAME, dtype=np.float32)
        # Quiet frames until the client has sent enough user audio.
        while self.received_samples < self.user_audio_s * SR and not ws.closed:
            await send_frame(silence)
        # The "reply": tone plus text pieces.
        n_reply = max(1, int(self.reply_s * SR / FRAME))
        for i in range(n_reply):
            if i < len(self.text):
                await ws.send_bytes(b"\x02" + self.text[i].encode("utf-8"))
            await send_frame(tone(FRAME))
        for _ in range(int(self.tail_s * SR / FRAME)):
            await send_frame(silence)
        if self.close_after:
            await ws.close()
        else:
            while not ws.closed:
                await send_frame(silence)
        recv_task.cancel()
        return ws

    async def start(self, port: int = 0) -> int:
        app = web.Application()
        app.router.add_get("/api/chat", self.handle_chat)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


async def _serve_forever(port: int) -> None:
    srv = MockServer(close_after=False)
    await srv.start(port)
    print(f"mock server on {port}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_serve_forever(int(sys.argv[1]) if len(sys.argv) > 1 else 8998))
