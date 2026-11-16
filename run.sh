#!/usr/bin/env bash
# Start the PersonaPlex server and the headless AVA voice client.
#
#   ./run.sh                    server + client (config.yaml)
#   ./run.sh --list-devices     print audio devices and exit
#   ./run.sh --no-server        connect to a moshi.server that is already running
#   ./run.sh --config my.yaml   use another config file
#
# Events (JSON lines) go to stdout, logs to stderr, server output to logs/server.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VPY="$ROOT/.venv/bin/python"

[[ -x "$VPY" ]] || { echo "run.sh: .venv not found. Run ./setup.sh first." >&2; exit 1; }

if [[ -f "$ROOT/.env" ]]; then
  set -a; # shellcheck disable=SC1091
  source "$ROOT/.env"; set +a
fi

case " $* " in
  *" --list-devices "*) ;;  # no token needed to list devices
  *) # HF_HUB_OFFLINE=1 uses the cached weights without contacting Hugging Face.
     [[ -n "${HF_TOKEN:-}" || "${HF_HUB_OFFLINE:-0}" == 1 ]] \
       || { echo "run.sh: HF_TOKEN is not set (see .env.example)." >&2; exit 1; } ;;
esac

cd "$ROOT"
exec "$VPY" -m ava_voice "$@"
