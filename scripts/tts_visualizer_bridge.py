#!/usr/bin/env python3
"""Bridge Hermes TTS output to a Windows desktop sound visualizer.

This script is intended for use from ``tts.visualizer.command``.  It copies the
final TTS audio file to a Windows machine reachable through the existing reverse
SSH tunnel, then asks Python on Windows to open an always-on-top bottom-right
visualizer window for that audio.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def run(cmd: list[str], timeout: float = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a TTS audio file to the main Windows PC visualizer")
    parser.add_argument("--audio", required=True, help="Local audio path produced by Hermes TTS")
    parser.add_argument("--text", default="", help="TTS text, shown as a small caption")
    parser.add_argument("--provider", default=os.environ.get("HERMES_TTS_PROVIDER", ""))
    parser.add_argument("--host", default=os.environ.get("MAINPC_SSH_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("MAINPC_SSH_PORT", "2222"))
    parser.add_argument("--user", default=os.environ.get("MAINPC_SSH_USER", "home"))
    parser.add_argument("--key", default=os.environ.get("MAINPC_SSH_KEY", "/home/ubuntu/.ssh/reverse_to_windows"))
    parser.add_argument("--remote-dir", default="C:/Users/home/AppData/Local/HermesTTSVisualizer")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists() or audio.stat().st_size <= 0:
        print(f"audio file missing or empty: {audio}", file=sys.stderr)
        return 2

    dest_name = f"tts_{uuid.uuid4().hex}{audio.suffix.lower() or '.ogg'}"
    remote_inbox = args.remote_dir.rstrip("/") + "/inbox"
    remote_audio = remote_inbox + "/" + dest_name
    target = f"{args.user}@{args.host}"
    ssh_base = [
        "ssh", "-p", str(args.port), "-i", str(Path(args.key).expanduser()),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new", target,
    ]
    scp_base = [
        "scp", "-P", str(args.port), "-i", str(Path(args.key).expanduser()),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=5",
        "-o", "StrictHostKeyChecking=accept-new",
    ]

    mkdir_cmd = f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"New-Item -ItemType Directory -Force -Path {ps_quote(remote_inbox)} | Out-Null\""
    created = run(ssh_base + [mkdir_cmd], timeout=args.timeout)
    if created.returncode != 0:
        print(created.stderr or created.stdout, file=sys.stderr)
        return created.returncode or 3

    copied = run(scp_base + [str(audio), f"{target}:{remote_audio}"], timeout=max(args.timeout, 20))
    if copied.returncode != 0:
        print(copied.stderr or copied.stdout, file=sys.stderr)
        return copied.returncode or 4

    text = args.text[:220]
    meta_name = f"{Path(dest_name).stem}.json"
    meta_payload = {
        "audio": remote_audio,
        "text": text,
        "provider": args.provider,
        "source": str(audio),
        "created_at": uuid.uuid4().hex,
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)
        local_meta = Path(fh.name)
    try:
        copied_meta = run(scp_base + [str(local_meta), f"{target}:{remote_inbox}/{meta_name}"], timeout=max(args.timeout, 20))
    finally:
        try:
            local_meta.unlink()
        except OSError:
            pass
    if copied_meta.returncode != 0:
        print(copied_meta.stderr or copied_meta.stdout, file=sys.stderr)
        return copied_meta.returncode or 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
