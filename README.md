# ava-voice

Lets the AVA robot hold spoken conversations through its own microphone and speaker,
using [NVIDIA PersonaPlex-7B](https://github.com/NVIDIA/personaplex), a full-duplex
speech-to-speech model. There is no browser: a headless Python client streams the mic
to a local PersonaPlex server and plays the reply. It also emits JSON events for the
robot's state machine and logs response latency.

> **Status: not yet tested on AVA or on any GPU machine.** The client, events, latency
> logging and server launcher are tested against a mock server (no GPU). The real
> model, `setup.sh` on Linux, real audio devices and the echo-cancel module have not
> been run yet. See [What has been tested](#what-has-been-tested).

## Quick start

On the robot's computer (Linux, NVIDIA GPU with at least 24 GB VRAM):

```bash
sudo apt install libopus-dev libportaudio2 git python3-venv
git clone <this repo> ava-voice && cd ava-voice
cp .env.example .env            # then put your Hugging Face token in .env
./setup.sh                      # venv, upstream at a pinned commit, ~16 GB of weights
.venv/bin/python scripts/check_hardware.py   # is this GPU fast enough?
./run.sh --list-devices         # find the mic and speaker, set them in config.yaml
./run.sh                        # talk to AVA. Ctrl-C to stop.
```

`HF_TOKEN` must come from a Hugging Face account that has accepted the NVIDIA Open
Model License on [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).
`setup.sh` stops with an explanation if libopus, PortAudio, a suitable Python (3.10 to
3.12), a CUDA GPU or Hugging Face access is missing.

`setup.sh` options: `--no-download` (do not prefetch weights), `--cpu-offload` (also
install `accelerate`). `run.sh` options: `--list-devices`, `--no-server` (connect to a
server you started yourself), `--config FILE`, `-v`.

Before using it with people, **edit `prompts/ava_role.txt`**. It is a draft.

## How it works

```
 mic ──► ava_voice client ──ws://127.0.0.1:8998/api/chat──► python -m moshi.server (GPU)
 speaker ◄──────────────┘ ◄── reply audio + reply text ───┘
            │
            ├─► stdout / local socket: JSON events (one per line)
            └─► logs/latency.csv
```

`run.sh` runs `python -m ava_voice`, which:

1. loads `config.yaml` and the role prompt,
2. starts `python -m moshi.server --host 127.0.0.1 --port 8998 --static none` (output in
   `logs/server.log`) and waits until it accepts connections (model load and warm-up),
3. connects, sends the voice and role prompt, then streams mic audio and plays the
   reply until stopped, reconnecting if the server drops the connection,
4. stops the server on exit.

Upstream does not document the websocket protocol. `ava_voice/protocol.py` describes
it as read from the upstream server and browser-client source at the pinned commit.
Summary: the query string carries `voice_prompt=NATF2.pt` and `text_prompt=...`. Every
message is binary, and its first byte gives the kind: `0x00` is the handshake (the
server sends it once the prompts are processed), `0x01` is Ogg/Opus audio (24 kHz mono,
both directions) and `0x02` is a UTF-8 piece of the model's reply text. If you bump the
upstream commit (`UPSTREAM_COMMIT` in `setup.sh`), `tests/test_protocol.py` checks those
facts against the new source.

## Config reference

`config.yaml` (relative paths are relative to the config file):

| Key | Default | Meaning |
|---|---|---|
| `server.launch` | `true` | Start `moshi.server` ourselves. `false` connects to one that is already running (same as `--no-server`). |
| `server.host` / `server.port` | `127.0.0.1` / `8998` | Where the client connects. |
| `server.cpu_offload` | `false` | Pass `--cpu-offload` (needs `./setup.sh --cpu-offload`). Much slower; not expected to be real time. |
| `server.startup_timeout_s` | `900` | How long to wait for model load and warm-up. |
| `server.log_file` | `logs/server.log` | Server stdout/stderr. |
| `server.extra_args` | `[]` | Extra `moshi.server` arguments, e.g. `["--max-pinned-frames", "768"]`. |
| `persona.voice` | `NATF2` | One of NATF0-3, NATM0-3, VARF0-4, VARM0-4. |
| `persona.role_prompt_file` | `prompts/ava_role.txt` | Role prompt. `#` lines are comments; the other lines are joined into one paragraph. English only. |
| `audio.input_device` / `audio.output_device` | `null` | `null` is the system default. Otherwise a device index or a name substring from `./run.sh --list-devices`. |
| `audio.playback_max_buffer_s` | `1.0` | Most reply audio queued for the speaker; older audio is dropped beyond this so delay cannot build up. |
| `echo.mode` | `none` | `none`, `external` or `pulse_echo_cancel`. See [Echo](#echo). |
| `echo.pulse.source_master` / `sink_master` | `null` | Real mic/speaker for `pulse_echo_cancel` (`pactl list short sources` / `sinks`). `null` uses the current defaults. |
| `echo.pulse.aec_method` / `aec_args` | `webrtc` / `""` | Passed to `module-echo-cancel`. |
| `events.stdout` | `true` | Write events to stdout. |
| `events.socket` | `null` | Also serve events on `unix:/path/to.sock` or `tcp:127.0.0.1:PORT`. |
| `detection.robot_speech_threshold_dbfs` | `-40` | Level above which the reply audio counts as the robot speaking. |
| `detection.robot_speech_hangover_s` | `0.5` | Quiet time before `robot_speaking_end`. |
| `detection.user_speech_threshold_dbfs` | `-40` | Mic level that counts as the user speaking (latency logging only). |
| `detection.user_speech_hangover_s` | `0.4` | Quiet time that ends a user utterance. |
| `latency.csv_path` | `logs/latency.csv` | `null` disables latency logging. |
| `web.enabled` / `web.host` / `web.port` | `true` / `127.0.0.1` / `8002` | Live conversation page. Local only by default (no login). |
| `web.history` | `300` | Events kept for a page that is opened later. |
| `asr.enabled` | `false` | Transcribe the user with a separate recognizer and emit `user_text`. |
| `asr.engine` / `asr.model_dir` / `asr.threads` | `parakeet` / `models` / `2` | Recognizer (Parakeet-TDT 0.6B v2, CPU), where its weights are, CPU threads. |
| `asr.min_speech_s` / `asr.max_speech_s` | `0.3` / `20` | Shorter utterances are ignored; longer ones are cut. |

The model always uses 24 kHz mono, and the client opens the devices at that rate. Desktop
audio servers (PulseAudio/PipeWire, which the ALSA `default`/`pulse`/`pipewire` devices go
through) resample automatically. Raw `hw:` ALSA devices often do not support 24 kHz.

## Events

Events are JSON objects, one per line, on **stdout** (logs go to stderr) and, if
`events.socket` is set, on every client connected to that socket. Socket clients only
receive; anything they send is ignored. Example consumer: `socat - UNIX-CONNECT:/tmp/ava-voice.sock`.

> **PersonaPlex does not produce a transcript of what the user said.** The only text it
> outputs is the robot's own reply. User text exists only if the optional ASR sidecar is
> enabled (`asr.enabled: true`, see [Web view and user transcript](#web-view-and-user-transcript)):
> then a separate recognizer (NVIDIA Parakeet-TDT) emits `user_text` events. That text is
> that recognizer's guess, not something PersonaPlex heard or used.

Every event has:

| Field | Type | Meaning |
|---|---|---|
| `v` | int | Schema version (currently `1`). |
| `type` | string | One of the types below. |
| `ts` | float | Unix time when the event was emitted. |
| `session` | string or null | ID of the current server connection. |

| `type` | Extra fields | When |
|---|---|---|
| `listening` | `reason`: `"session_start"` or `"robot_finished"` | The conversation is live and the robot is not speaking. The mic is streamed all the time, including while the robot speaks. |
| `robot_speaking_start` | `turn` (int), `audible_at` (float, estimated Unix time the audio reaches the speaker), `latency_ms` (int or null, see below) | Reply audio rose above the threshold. Emitted when the audio arrives, slightly before it is heard. |
| `robot_text` | `turn` (int), `delta` (string, may start with a space) | A piece of the robot's reply text, as the model generates it. It may arrive just before the matching `robot_speaking_start`, and then carries that upcoming turn number. |
| `robot_speaking_end` | `turn` (int), `audible_until` (float), `text` (string, the turn's full reply text) | Reply audio stayed below the threshold for `robot_speech_hangover_s`. |
| `session_closed` | `reason`: `"server_closed"`, `"stopped"` or `"error"` | The connection ended. A new `listening` follows if the client reconnects. |
| `user_text` | `text`, `audio_s` (utterance length), `asr_ms` (transcription time), `engine`, `robot_was_speaking` (bool) | **Only with `asr.enabled: true`.** One finished user utterance, transcribed by the separate recognizer. Arrives `asr_ms` after the utterance ended; may come after the robot already started answering. |

Example:

```json
{"v": 1, "type": "listening", "ts": 1763290000.123, "session": "3f9c2a1b7d4e", "reason": "session_start"}
{"v": 1, "type": "robot_text", "ts": 1763290003.401, "session": "3f9c2a1b7d4e", "turn": 1, "delta": " Hi"}
{"v": 1, "type": "robot_speaking_start", "ts": 1763290003.420, "session": "3f9c2a1b7d4e", "turn": 1, "audible_at": 1763290003.51, "latency_ms": 640}
{"v": 1, "type": "robot_speaking_end", "ts": 1763290006.010, "session": "3f9c2a1b7d4e", "turn": 1, "audible_until": 1763290005.6, "text": "Hi, I'm AVA."}
{"v": 1, "type": "listening", "ts": 1763290006.010, "session": "3f9c2a1b7d4e", "reason": "robot_finished"}
```

Speaking start and end are inferred from the loudness of the reply audio, because the
server streams audio all the time, with near-silence when the robot is not talking.
Breaths or very quiet speech can shift the boundaries, so tune the `detection.*` values on
the robot.

## Web view and user transcript

While `./run.sh` runs, open **http://127.0.0.1:8002** (from a laptop:
`ssh -L 8002:localhost:8002 <user>@<robot>`, then http://localhost:8002). The page shows:

- the state (listening / AVA speaking / disconnected),
- the conversation: AVA's reply text as it is generated, and, with the ASR sidecar, what
  the user said (labelled with the recognizer that produced it),
- response latency per turn from `logs/latency.csv` (median, 95th percentile, last),
- the raw event stream.

The page has no login and listens on 127.0.0.1 only. It reads `GET /events`
(Server-Sent Events), `/api/history` and `/api/latency`, so other tools can use those too.

To show what the user said, enable the ASR sidecar:

```bash
.venv/bin/pip install "sherpa-onnx>=1.12"
.venv/bin/python -m ava_voice.asr --download     # ~0.5 GB into models/, SHA-256 checked
# config.yaml: asr: {enabled: true}
```

The client cuts the microphone stream into utterances with its speech detector (with 0.3 s
of pre-roll) and transcribes each finished one on the CPU (2 threads by default), one at a
time. This never changes the audio sent to PersonaPlex. Parakeet-TDT is the default because it
was the most accurate and by far the fastest of three recognizers (Vosk, Whisper small.en,
Parakeet) in a first small test on a laptop (20 LibriSpeech utterances, `ava-speech-eval`
repository); the full benchmark on the robot computer is still to be run. Without echo cancellation it will also transcribe AVA's own voice; such
utterances are marked `robot_was_speaking: true`.

## Latency log

`logs/latency.csv` gets one row per robot turn that follows user speech:

```
session,turn,user_speech_end_unix,robot_audio_start_unix,latency_ms
```

- `user_speech_end_unix`: the end of the last mic segment above
  `user_speech_threshold_dbfs`, corrected for the input device latency.
- `robot_audio_start_unix`: when the first loud reply audio is expected to leave the
  speaker. This is the arrival time plus the audio already queued plus the output
  device latency reported by PortAudio.

No row is written if the robot starts while the user is still talking (an overlap or a
backchannel), or if no user speech came before the turn. Mic activity while the robot is
talking is ignored for this metric, because without echo cancellation it is mostly the
robot's own voice. The measurement uses a simple level detector, so background noise
above the threshold delays the detected end of speech. Without echo cancellation, the
numbers are unreliable.

## Echo

The robot's mic hears its own speaker. PersonaPlex is full duplex: it listens while it
talks, which is how users can interrupt it. If it hears itself, it may stop mid-sentence
or answer its own voice. **ava-voice never mutes the mic while the robot talks**, because
that would make interruption impossible. The echo has to be removed before the audio
reaches the client. Choose one approach with `echo.mode`:

1. **Headset or echo-cancelling speakerphone** (`echo.mode: external`). This is the most
   reliable option. A USB conference speakerphone (with built-in AEC) works well on a
   mobile robot. Select it as both input and output device.
2. **PulseAudio/PipeWire echo cancellation, managed by ava-voice** (`echo.mode:
   pulse_echo_cancel`). On start, the client runs
   `pactl load-module module-echo-cancel aec_method=webrtc source_name=ava_ec_source sink_name=ava_ec_sink [source_master=...] [sink_master=...]`,
   points its streams at those virtual devices (it sets `PULSE_SOURCE`/`PULSE_SINK` and uses
   the `pulse` device unless you set devices explicitly), and unloads the module on exit.
   This needs `pactl` (package `pulseaudio-utils`, which also works with PipeWire through
   `pipewire-pulse`). Set `echo.pulse.source_master`/`sink_master` if the robot's mic and
   speaker are not the system defaults.
3. **System-wide echo cancellation** (`echo.mode: external`). Configure echo-cancel
   permanently in the audio server and select the resulting devices in `config.yaml`. For
   PulseAudio, add `load-module module-echo-cancel ...` to `/etc/pulse/default.pa`. For
   PipeWire, add a `libpipewire-module-echo-cancel` entry under
   `~/.config/pipewire/pipewire.conf.d/`.

With `echo.mode: none`, a warning is printed at start-up and nothing else changes.

Software AEC needs the speaker signal it cancels to be the exact signal that is played. It
works best at moderate volume, without other audio playing, and with the mic not right
next to the speaker. Neither software option has been tried on AVA yet.

## Hardware check

```bash
.venv/bin/python scripts/check_hardware.py [--input-wav clip.wav] [--voice NATM1] [--cpu-offload]
```

This prints the GPU name and VRAM and runs one clip (by default upstream's
`assets/test/input_assistant.wav`) through upstream's offline inference. That uses the same
encode, model step and decode loop the server runs every 80 ms. It then reports:

- per-frame time (mean and 95th percentile) against the 80 ms budget,
- **real-time factor** (RTF): processing time divided by audio duration, for the streaming
  loop only (model loading is excluded),
- peak GPU memory (allocated and reserved by PyTorch).

Exit code `0` means real time (with a warning if RTF > 0.8 or the 95th percentile exceeds
80 ms). Exit code `1` means too slow (RTF ≥ 1.0), out of memory or failed. Exit code `2`
means no CUDA GPU or no PyTorch.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `setup.sh: libopus was not found` | `sudo apt install libopus-dev` |
| `PortAudio was not found` | `sudo apt install libportaudio2` |
| `HF_TOKEN is not set` | `cp .env.example .env` and fill it in. |
| `Hugging Face rejected HF_TOKEN (401)` | The token is wrong or revoked; create a new one. |
| `...has not been granted access...` | Accept the license on the model page while logged in as the token's owner. |
| `PyTorch cannot see a CUDA GPU` | Check `nvidia-smi`. On Blackwell GPUs, upstream suggests `.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`. This goes outside upstream's `torch<2.5` pin and is untested here. |
| `server exited ... before it was ready` | Read `logs/server.log`. Out-of-memory means the GPU is too small: try `server.cpu_offload: true` (slow). |
| `cannot open audio devices at 24000 Hz` | Pick a `pulse`, `pipewire` or `default` device instead of a raw `hw:` device. |
| `no handshake within 120s` | Another client (e.g. a browser) holds the server; it serves one conversation at a time. |
| Robot cuts itself off or talks to itself | Echo. See [Echo](#echo). |
| `robot_speaking_start/end` flicker or never fire | Adjust `detection.robot_speech_threshold_dbfs` / `hangover_s`. Run with `-v` and watch the events. |
| Replies drift or ignore the role | Shorten the role prompt. Voice plus text must fit in `--max-pinned-frames` (512 by default; `server.log` prints `Pinned N system prompt frames`). Conversation memory is about the last 2.5 minutes. |
| No internet on the robot | After weights are cached, run with `HF_HUB_OFFLINE=1 ./run.sh`. |

**Security:** upstream starts its web server without a bind address, so it listens on
**all network interfaces** on port 8998, whatever `--host` says. Anyone on the network
could connect and talk to the model, or block the robot's session. Firewall port 8998 on
the robot if its network is not trusted (e.g. `sudo ufw deny 8998`; local connections are
unaffected). TLS is not used: the client is local, and upstream's `--ssl` would download
`mkcert` and install a local root CA.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests need no GPU, model, network or audio hardware. A mock server
(`tests/mock_server.py`) follows the upstream handler's behaviour, and fake audio devices
replace the mic and speaker. Together they check the full client: the handshake comes
before any audio, the Opus round-trip works, the robot speaking/text events and session
close are emitted, latency is logged, the mic is never muted, and the launcher starts the
server, waits for it and shuts it down. `tests/test_protocol.py` also checks the protocol
facts against the upstream source when `third_party/personaplex` exists.

## What has been tested

Run on a development Mac (no NVIDIA GPU):

- the test suite above (47 tests, all passing; the upstream-source check ran against a clone of the pinned commit),
- upstream `moshi-personaplex` at the pinned commit installed next to ava-voice in one
  venv, with no version conflicts. The upstream functions that `check_hardware.py` hooks
  into exist with the expected names and signature,
- `./run.sh --list-devices` with real PortAudio, plus the `run.sh`/`setup.sh` errors for a
  missing token and missing libopus, and `ava_voice.hf` with a missing or invalid token.

**Not yet run:** `setup.sh` from start to finish on Linux; the real PersonaPlex server and
model; `check_hardware.py` on a GPU; real mic/speaker streaming; `pulse_echo_cancel`; the
Hugging Face path for a valid token, both with and without the license accepted; anything
on AVA.

## License notes

- **This repository** (wrapper code): MIT, see `LICENSE`.
- **PersonaPlex code** ([NVIDIA/personaplex](https://github.com/NVIDIA/personaplex)): MIT
  (with Kyutai Moshi and AudioCraft notices in its `moshi/` directory). `setup.sh` clones it
  into `third_party/`, which is gitignored. None of it is copied into this repository.
- **PersonaPlex weights** (`nvidia/personaplex-7b-v1`): NVIDIA Open Model License, gated on
  Hugging Face. They are downloaded to the Hugging Face cache on the robot, never committed.
  Read the license before deploying.
- Never commit `.env` or any token. `.env` is gitignored, and `.env.example` holds no secrets.
