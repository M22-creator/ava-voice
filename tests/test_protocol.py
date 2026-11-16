import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ava_voice import protocol


def test_build_url_has_both_required_params_and_no_seed():
    url = protocol.build_url("127.0.0.1", 8998, "NATF2", "You are AVA. Be brief & kind?")
    u = urlparse(url)
    assert u.scheme == "ws" and u.netloc == "127.0.0.1:8998" and u.path == "/api/chat"
    q = parse_qs(u.query, keep_blank_values=True)
    assert q == {"voice_prompt": ["NATF2.pt"], "text_prompt": ["You are AVA. Be brief & kind?"]}


def test_build_url_empty_prompt_is_still_sent():
    q = parse_qs(urlparse(protocol.build_url("h", 1, "NATM0", "")).query, keep_blank_values=True)
    assert q["text_prompt"] == [""]


def test_ssl_scheme():
    assert protocol.build_url("h", 1, "NATF0", "x", use_ssl=True).startswith("wss://")


def test_encode_decode():
    assert protocol.encode_audio(b"abc") == b"\x01abc"
    assert protocol.decode(b"\x02 hi") == (2, b" hi")
    assert protocol.decode(b"\x00") == (0, b"")
    with pytest.raises(protocol.ProtocolError):
        protocol.decode(b"")


def test_voices_match_upstream_list():
    assert len(protocol.VOICES) == 18
    assert {"NATF0", "NATF3", "NATM3", "VARF4", "VARM4"} <= set(protocol.VOICES)
    assert protocol.FRAME_SIZE == 1920


# Guard for bumping the pinned upstream commit: re-check the facts protocol.py
# relies on against the upstream server source. Skipped when the upstream
# checkout is not present (e.g. before ./setup.sh).
UPSTREAM = Path(os.environ.get("AVA_UPSTREAM_DIR", Path(__file__).parent.parent / "third_party/personaplex"))
SERVER_PY = UPSTREAM / "moshi/moshi/server.py"


@pytest.mark.skipif(not SERVER_PY.exists(), reason="upstream checkout not present")
def test_upstream_server_still_matches_protocol_assumptions():
    src = SERVER_PY.read_text()
    assert 'add_get("/api/chat"' in src
    assert 'request.query["voice_prompt"]' in src
    assert 'request.query["text_prompt"]' in src
    assert 'send_bytes(b"\\x00")' in src  # handshake
    assert 'b"\\x01" + msg' in src  # audio to client
    assert 'b"\\x02" + bytes(_text' in src  # text to client
    assert re.search(r"if kind == 1:\s+# audio", src)  # only audio accepted from client
    assert "sphn.OpusStreamReader(self.mimi.sample_rate)" in src
