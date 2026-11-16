#!/usr/bin/env bash
# Set up ava-voice: venv, upstream PersonaPlex at a pinned commit, dependency checks.
#
#   ./setup.sh                 full setup, including downloading the weights (~16 GB)
#   ./setup.sh --no-download   skip prefetching weights (the server downloads them on first run)
#   ./setup.sh --cpu-offload   also install `accelerate` for server.cpu_offload
#
# Environment: PYTHON=/path/to/python3.x to choose the interpreter (3.10-3.12).
set -euo pipefail

UPSTREAM_URL="https://github.com/NVIDIA/personaplex.git"
# Pinned upstream commit. The websocket protocol in ava_voice/protocol.py was read from this
# exact revision; re-check it before bumping.
UPSTREAM_COMMIT="16419a2ef7b98611f58c9df07477d00e5ad61d5f"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
UPSTREAM_DIR="$ROOT/third_party/personaplex"

DOWNLOAD=1
CPU_OFFLOAD=0
for arg in "$@"; do
  case "$arg" in
    --no-download) DOWNLOAD=0 ;;
    --cpu-offload) CPU_OFFLOAD=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { echo "==> $*"; }
fail() { echo; echo "setup.sh: ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Checks that need no downloads. Fail fast with a clear fix.
# ---------------------------------------------------------------------------
[[ "$(uname -s)" == "Linux" ]] || echo "WARNING: ava-voice targets Linux with an NVIDIA GPU; $(uname -s) is untested." >&2

command -v git >/dev/null || fail "git is not installed. Fix: sudo apt install git"

have_lib() { ldconfig -p 2>/dev/null | grep -q "$1" ; }

if ! { pkg-config --exists opus 2>/dev/null || have_lib 'libopus\.so'; }; then
  fail "libopus was not found. PersonaPlex needs the Opus codec library.
  Fix: sudo apt install libopus-dev"
fi
if ! have_lib 'libportaudio\.so'; then
  fail "PortAudio was not found. The client uses it (via python-sounddevice) for the mic and speaker.
  Fix: sudo apt install libportaudio2"
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$ROOT/.env"; set +a
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN is not set. The PersonaPlex weights are gated on Hugging Face.
  Fix: 1) log in at https://huggingface.co/nvidia/personaplex-7b-v1 and accept the license
       2) create a read token at https://huggingface.co/settings/tokens
       3) cp .env.example .env   and put the token after HF_TOKEN="
fi
export HF_TOKEN

if ! command -v nvidia-smi >/dev/null; then
  echo "WARNING: nvidia-smi not found; the server needs an NVIDIA GPU and driver." >&2
fi

# ---------------------------------------------------------------------------
# 2. Python venv. Upstream pins torch<2.5, which has wheels for Python 3.10-3.12.
# ---------------------------------------------------------------------------
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; return; fi
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))'; then
      echo "$c"; return
    fi
  done
}
PY="$(pick_python)"
[[ -n "$PY" ]] || fail "no Python 3.10-3.12 found (upstream needs torch<2.5).
  Fix: sudo apt install python3.12 python3.12-venv   (or set PYTHON=/path/to/python3.x)"
"$PY" -c 'import venv, ensurepip' 2>/dev/null || fail "$PY cannot create venvs.
  Fix: sudo apt install python3-venv   (matching your Python version, e.g. python3.12-venv)"

if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv in .venv with $("$PY" --version)"
  "$PY" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# 3. Upstream PersonaPlex at the pinned commit (never committed to this repo).
# ---------------------------------------------------------------------------
if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
  say "cloning $UPSTREAM_URL into third_party/"
  mkdir -p "$ROOT/third_party"
  git clone --quiet "$UPSTREAM_URL" "$UPSTREAM_DIR"
fi
if ! git -C "$UPSTREAM_DIR" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
  say "fetching upstream"
  git -C "$UPSTREAM_DIR" fetch --quiet origin
fi
git -C "$UPSTREAM_DIR" -c advice.detachedHead=false checkout --quiet "$UPSTREAM_COMMIT" \
  || fail "could not check out upstream commit $UPSTREAM_COMMIT (local changes in third_party/personaplex?)"
say "upstream at $(git -C "$UPSTREAM_DIR" rev-parse --short HEAD)"

say "installing upstream moshi (PersonaPlex) - this pulls PyTorch and can take a while"
"$VPY" -m pip install --quiet "$UPSTREAM_DIR/moshi/"
say "installing ava-voice"
"$VPY" -m pip install --quiet -e "$ROOT[test]"
if [[ "$CPU_OFFLOAD" == 1 ]]; then
  "$VPY" -m pip install --quiet accelerate
fi

"$VPY" - <<'EOF' || fail "PyTorch cannot see a CUDA GPU. Check the NVIDIA driver (nvidia-smi). Blackwell GPUs need a newer
  torch build, see README > Troubleshooting."
import torch, sys
ok = torch.cuda.is_available()
print("==> torch", torch.__version__, "CUDA", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0) if ok else "none")
sys.exit(0 if ok else 1)
EOF

# ---------------------------------------------------------------------------
# 4. Hugging Face access (and weights).
# ---------------------------------------------------------------------------
say "checking Hugging Face access to nvidia/personaplex-7b-v1"
if [[ "$DOWNLOAD" == 1 ]]; then
  "$VPY" -m ava_voice.hf --download || fail "Hugging Face check failed (message above)."
else
  "$VPY" -m ava_voice.hf || fail "Hugging Face check failed (message above)."
fi

mkdir -p "$ROOT/logs"
echo
say "setup complete."
echo "    Next:  .venv/bin/python scripts/check_hardware.py   (is this GPU fast enough?)"
echo "           ./run.sh --list-devices                      (pick mic/speaker in config.yaml)"
echo "           ./run.sh"
