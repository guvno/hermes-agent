#!/usr/bin/env python3
"""Bridge Hermes TTS output to a Windows desktop sound visualizer.

This script is intended for use from ``tts.visualizer.command``.  It copies the
final TTS audio file to a Windows machine reachable through the existing reverse
SSH tunnel, then asks Python on Windows to open an always-on-top bottom-right
visualizer window for that audio.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
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
    parser.add_argument("--remote-script", default="C:/Users/home/AppData/Local/HermesTTSVisualizer/tts_visualizer_win.py")
    parser.add_argument("--python", default="python")
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
    ps = (
        "$p = Start-Process -WindowStyle Hidden -PassThru "
        f"-FilePath {ps_quote(args.python)} "
        f"-ArgumentList @({ps_quote(args.remote_script)}, '--audio', {ps_quote(remote_audio)}, '--text', {ps_quote(text)}, '--provider', {ps_quote(args.provider)})"
    )
    launched = run(
        ssh_base + [f"powershell -NoProfile -ExecutionPolicy Bypass -Command {shlex.quote(ps)}"],
        timeout=args.timeout,
    )
    if launched.returncode != 0:
        print(launched.stderr or launched.stdout, file=sys.stderr)
        return launched.returncode or 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
