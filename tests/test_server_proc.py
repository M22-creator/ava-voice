import asyncio
import socket
import sys

import pytest

from ava_voice.config import ServerConfig
from ava_voice.server_proc import ServerError, ServerProcess, build_server_argv


def test_argv():
    argv = build_server_argv(ServerConfig(port=9000, cpu_offload=True, extra_args=["--max-pinned-frames", 600]), python="py")
    assert argv[:3] == ["py", "-m", "moshi.server"]
    assert argv[argv.index("--port") + 1] == "9000"
    assert argv[argv.index("--static") + 1] == "none"
    assert "--cpu-offload" in argv and "--ssl" not in argv
    assert argv[-2:] == ["--max-pinned-frames", "600"]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_waits_until_port_opens(tmp_path):
    port = free_port()
    code = (
        "import socket, time; time.sleep(1.0); s = socket.socket(); "
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        f"s.bind(('127.0.0.1', {port})); s.listen(); print('up', flush=True); time.sleep(30)"
    )
    srv = ServerProcess([sys.executable, "-c", code], tmp_path / "logs/server.log")
    srv.start()
    try:
        asyncio.run(srv.wait_ready("127.0.0.1", port, timeout_s=20))
    finally:
        srv.stop()
    assert srv.proc.returncode is not None
    assert "up" in (tmp_path / "logs/server.log").read_text()


def test_early_exit_reports_log_tail(tmp_path):
    code = "import sys; print('torch.cuda.OutOfMemoryError: boom'); sys.exit(3)"
    srv = ServerProcess([sys.executable, "-c", code], tmp_path / "server.log")
    srv.start()
    with pytest.raises(ServerError, match="code 3[\\s\\S]*OutOfMemoryError"):
        asyncio.run(srv.wait_ready("127.0.0.1", free_port(), timeout_s=20))
    srv.stop()
