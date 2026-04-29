#!/usr/bin/env python3
"""Resident Windows listener for Hermes TTS visualizer inbox.

Runs in the interactive Windows logon session (preferably via Scheduled Task).
It watches the inbox for metadata JSON files written by the server bridge and
launches the Tk visualizer from the same desktop session, avoiding SSH session
isolation where GUI windows may be invisible.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AUDIO_SUFFIXES = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}
DEFAULT_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "HermesTTSVisualizer"
DEFAULT_INBOX = DEFAULT_BASE / "inbox"
DEFAULT_SCRIPT = DEFAULT_BASE / "tts_visualizer_win.py"
LOG_PATH = DEFAULT_BASE / "listener.log"
SEEN_PATH = DEFAULT_BASE / "listener_seen.json"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def load_seen() -> dict[str, float]:
    try:
        data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_seen(seen: dict[str, float]) -> None:
    # Keep this small; old files in inbox should not grow state forever.
    cutoff = time.time() - 7 * 24 * 3600
    compact = {k: v for k, v in seen.items() if v >= cutoff}
    SEEN_PATH.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")


def stable(path: Path, delay: float = 0.25) -> bool:
    try:
        size1 = path.stat().st_size
        time.sleep(delay)
        size2 = path.stat().st_size
        return size1 > 0 and size1 == size2
    except OSError:
        return False


def launch_visualizer(python_exe: str, visualizer_script: Path, audio: Path, text: str = "", provider: str = "") -> bool:
    if not visualizer_script.exists():
        log(f"visualizer script missing: {visualizer_script}")
        return False
    if not audio.exists():
        log(f"audio missing: {audio}")
        return False
    cmd = [python_exe, str(visualizer_script), "--audio", str(audio), "--text", text or "", "--provider", provider or ""]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        log(f"launched visualizer for {audio.name}")
        return True
    except Exception as exc:
        log(f"launch failed for {audio}: {exc!r}")
        return False


def handle_meta(meta: Path, python_exe: str, visualizer_script: Path) -> bool:
    if not stable(meta):
        return False
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"metadata parse failed {meta.name}: {exc!r}")
        return False
    audio_value = payload.get("audio") or payload.get("audio_path") or payload.get("path")
    if not audio_value:
        log(f"metadata has no audio path: {meta.name}")
        return False
    audio = Path(str(audio_value))
    if not audio.is_absolute():
        audio = meta.parent / audio
    if not stable(audio):
        log(f"audio not stable yet: {audio}")
        return False
    text = str(payload.get("text") or "")[:220]
    provider = str(payload.get("provider") or "")[:80]
    return launch_visualizer(python_exe, visualizer_script, audio, text=text, provider=provider)


def handle_audio(audio: Path, python_exe: str, visualizer_script: Path) -> bool:
    if not stable(audio):
        return False
    return launch_visualizer(python_exe, visualizer_script, audio, text=audio.name, provider="")


def run_once(inbox: Path, python_exe: str, visualizer_script: Path, seen: dict[str, float], audio_fallback: bool = False) -> int:
    inbox.mkdir(parents=True, exist_ok=True)
    launched = 0
    # Metadata files are the normal bridge contract. Audio-only fallback is opt-in
    # for manual tests; otherwise the listener can race the bridge and launch once
    # for the raw audio and again when metadata arrives.
    candidates = sorted(inbox.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if audio_fallback:
        candidates += sorted((p for p in inbox.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES), key=lambda p: p.stat().st_mtime)
    for path in candidates:
        key = str(path.resolve())
        mtime = path.stat().st_mtime
        if seen.get(key) == mtime:
            continue
        ok = handle_meta(path, python_exe, visualizer_script) if path.suffix.lower() == ".json" else handle_audio(path, python_exe, visualizer_script)
        if ok:
            seen[key] = mtime
            launched += 1
        else:
            # Mark very old bad items as seen so the log does not get carpet-bombed.
            if time.time() - mtime > 3600:
                seen[key] = mtime
    return launched


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch Hermes TTS visualizer inbox and launch visible desktop windows")
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX))
    parser.add_argument("--visualizer-script", default=str(DEFAULT_SCRIPT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--interval", type=float, default=0.7)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--audio-fallback", action="store_true", help="Also launch for bare audio files without metadata")
    args = parser.parse_args()

    inbox = Path(args.inbox)
    visualizer_script = Path(args.visualizer_script)
    DEFAULT_BASE.mkdir(parents=True, exist_ok=True)
    seen = load_seen()
    log(f"listener start inbox={inbox} python={args.python} script={visualizer_script} once={args.once}")
    while True:
        try:
            launched = run_once(inbox, args.python, visualizer_script, seen, audio_fallback=args.audio_fallback)
            if launched:
                save_seen(seen)
        except Exception as exc:
            log(f"listener loop error: {exc!r}")
        if args.once:
            save_seen(seen)
            return 0
        time.sleep(max(0.2, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
