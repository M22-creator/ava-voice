"""PersonaPlex websocket protocol.

Upstream does not document this protocol. Everything here was read from the
upstream source at the commit pinned in ``setup.sh`` (UPSTREAM_COMMIT):

* ``moshi/moshi/server.py``  -- ``ServerState.handle_chat``
* ``client/src/protocol/encoder.ts`` and ``types.ts``  -- the browser client
* ``client/src/pages/Conversation/Conversation.tsx``  -- how the URL is built
* ``client/src/pages/Conversation/hooks/useUserAudio.ts``  -- mic encoding

Facts we rely on:

* Endpoint: ``GET /api/chat`` upgraded to a websocket.
* Query parameters read by the server: ``voice_prompt`` (a file name inside
  the server's voice prompt directory, e.g. ``NATF2.pt``) and ``text_prompt``
  (may be empty). Both keys must be present: the handler indexes
  ``request.query[...]`` directly and fails with KeyError otherwise.
  The browser also sends sampling parameters, but the server code that would
  read them is commented out, so we do not send them.
* We deliberately do NOT send ``seed``: the handler reads it with
  ``request["seed"]`` (not ``request.query``), which raises KeyError.
* Every websocket message is binary; the first byte is the kind:

  - ``0x00`` handshake, server -> client, sent once after the voice and text
    prompts have been processed. The client must not send audio before it.
  - ``0x01`` audio, both directions. Payload is a chunk of an Ogg/Opus
    stream, mono, 24 kHz (``sphn.OpusStreamWriter(24000)`` on the server; the
    browser uses opus-recorder with encoderSampleRate 24000).
  - ``0x02`` text, server -> client. Payload is one UTF-8 text piece of the
    model's *own* reply (a SentencePiece token with the ``▁`` word marker
    replaced by a space). There is no transcript of the user's speech.

  The browser client also defines kinds 0x03-0x06 (control, metadata, error,
  ping), but the server never sends them and ignores anything other than
  0x01 from the client, so we do not implement them.
* The server handles one conversation at a time (``asyncio.Lock``); a second
  connection waits until the first one closes.
"""

from __future__ import annotations

from urllib.parse import urlencode

MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02

SAMPLE_RATE = 24000  # moshi/moshi/models/loaders.py: SAMPLE_RATE
FRAME_RATE = 12.5  # moshi/moshi/models/loaders.py: FRAME_RATE
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples = 80 ms

# Voices shipped in voices.tgz of nvidia/personaplex-7b-v1 (upstream README).
VOICES = tuple(
    [f"NATF{i}" for i in range(4)]
    + [f"NATM{i}" for i in range(4)]
    + [f"VARF{i}" for i in range(5)]
    + [f"VARM{i}" for i in range(5)]
)


class ProtocolError(Exception):
    pass


def build_url(host: str, port: int, voice: str, text_prompt: str, use_ssl: bool = False) -> str:
    scheme = "wss" if use_ssl else "ws"
    query = urlencode({"voice_prompt": f"{voice}.pt", "text_prompt": text_prompt})
    return f"{scheme}://{host}:{port}/api/chat?{query}"


def encode_audio(opus_bytes: bytes) -> bytes:
    return bytes([MSG_AUDIO]) + opus_bytes


def decode(message: bytes) -> tuple[int, bytes]:
    """Split a server message into (kind, payload)."""
    if not message:
        raise ProtocolError("empty message")
    return message[0], message[1:]
