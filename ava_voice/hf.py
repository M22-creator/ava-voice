"""Check Hugging Face access to the gated PersonaPlex weights and optionally prefetch them.

Used by setup.sh:  python -m ava_voice.hf [--download]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = "nvidia/personaplex-7b-v1"
# File names from upstream moshi/moshi/models/loaders.py and server.py at the pinned commit.
FILES = [
    "config.json",
    "tokenizer_spm_32k_3.model",
    "tokenizer-e351c8d8-checkpoint125.safetensors",
    "voices.tgz",
    "model.safetensors",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="download all weight files into the HF cache")
    args = p.parse_args(argv)

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set (put it in .env, see .env.example).", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    # An invalid token also shows up as GatedRepoError below, so check the token itself first.
    import requests

    try:
        HfApi().whoami()
    except requests.HTTPError as e:  # hub 0.24 re-raises whoami failures as plain HTTPError
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 401 or "Invalid user token" in str(e):
            print("Hugging Face rejected HF_TOKEN (401). Create a new read token at "
                  "https://huggingface.co/settings/tokens and update .env.", file=sys.stderr)
        else:
            print(f"Could not check HF_TOKEN with Hugging Face ({e}).", file=sys.stderr)
        return 1
    except requests.ConnectionError as e:
        print(f"Could not reach huggingface.co ({e}). Check the network.", file=sys.stderr)
        return 1

    try:
        hf_hub_download(REPO, "config.json")
    except GatedRepoError:
        print(
            f"HF_TOKEN works, but this Hugging Face account has not been granted access to {REPO}.\n"
            f"Open https://huggingface.co/{REPO} while logged in as the token's owner, accept the\n"
            "NVIDIA Open Model License, then run ./setup.sh again.",
            file=sys.stderr,
        )
        return 1
    except RepositoryNotFoundError:
        print(f"Hugging Face says {REPO} does not exist or HF_TOKEN is invalid.", file=sys.stderr)
        return 1
    except HfHubHTTPError as e:
        print(f"Could not reach Hugging Face ({e}). Check HF_TOKEN and network access.", file=sys.stderr)
        return 1
    print(f"Hugging Face access to {REPO}: OK")

    if args.download:
        for name in FILES:
            print(f"fetching {name} ...", flush=True)
            hf_hub_download(REPO, name)
        print("weights cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
