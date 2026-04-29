#!/usr/bin/env python3
"""Always-on-top corner visualizer for Hermes TTS audio on Windows.

No third-party Python packages are required.  ffmpeg is used when available so
Telegram-friendly OGG/Opus files work; WAV files fall back to the stdlib wave
module.
"""
from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from tkinter import BOTH, Canvas, Tk

SAMPLE_RATE = 48_000
FPS = 30
BAR_COUNT = 48
WINDOW_W = 460
WINDOW_H = 150
MARGIN = 16


def decode_with_ffmpeg(path: Path) -> list[int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return []
    count = len(proc.stdout) // 2
    return list(struct.unpack("<" + "h" * count, proc.stdout[: count * 2]))


def decode_wave(path: Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        return [], sample_rate
    values = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels <= 1:
        return list(values), sample_rate
    return [int(sum(values[i : i + channels]) / channels) for i in range(0, len(values), channels)], sample_rate


def load_envelope(path: Path) -> tuple[list[list[float]], float]:
    samples = decode_with_ffmpeg(path)
    sample_rate = SAMPLE_RATE
    if not samples and path.suffix.lower() == ".wav":
        samples, sample_rate = decode_wave(path)
    if not samples:
        # A graceful failure still gives the user a visible pulse rather than a
        # dead square in the corner. Technology must at least look guilty.
        frames = [[0.15 + 0.1 * math.sin(i / 3 + b / 4) for b in range(BAR_COUNT)] for i in range(FPS * 3)]
        return frames, 3.0

    samples_per_frame = max(1, int(sample_rate / FPS))
    frames: list[list[float]] = []
    for start in range(0, len(samples), samples_per_frame):
        chunk = samples[start : start + samples_per_frame]
        if not chunk:
            continue
        band_size = max(1, len(chunk) // BAR_COUNT)
        bars: list[float] = []
        for b in range(BAR_COUNT):
            band = chunk[b * band_size : (b + 1) * band_size]
            if not band:
                bars.append(0.0)
                continue
            rms = math.sqrt(sum(x * x for x in band) / len(band)) / 32768.0
            bars.append(min(1.0, rms * 4.8))
        frames.append(bars)
    duration = len(samples) / sample_rate
    return frames or [[0.0] * BAR_COUNT], max(0.8, duration)


class Visualizer:
    def __init__(self, audio: Path, text: str, provider: str):
        self.audio = audio
        self.text = text
        self.provider = provider
        self.frames: list[list[float]] = [[0.0] * BAR_COUNT]
        self.duration = 2.0
        self.index = 0
        self.root = Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.92)
        except Exception:
            pass
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(0, screen_w - WINDOW_W - 18)
        y = max(0, screen_h - WINDOW_H - 58)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")
        self.canvas = Canvas(self.root, width=WINDOW_W, height=WINDOW_H, bg="#05070d", bd=0, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        self.frames, self.duration = load_envelope(self.audio)

    def draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, WINDOW_W, WINDOW_H, fill="#05070d", outline="#152033")
        self.canvas.create_text(MARGIN, 15, anchor="w", text="HERMES TTS", fill="#7dd3fc", font=("Segoe UI", 9, "bold"))
        caption = self.text.replace("\n", " ")[:72] or self.audio.name
        self.canvas.create_text(MARGIN + 92, 15, anchor="w", text=caption, fill="#cbd5e1", font=("Segoe UI", 8))
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        usable_w = WINDOW_W - MARGIN * 2
        bar_w = usable_w / BAR_COUNT
        base_y = WINDOW_H - 25
        for i, amp in enumerate(frame):
            eased = min(1.0, amp ** 0.55)
            h = 7 + eased * 88
            x0 = MARGIN + i * bar_w + 1
            x1 = MARGIN + (i + 1) * bar_w - 1
            y0 = base_y - h
            color = "#22d3ee" if i % 3 else "#a78bfa"
            self.canvas.create_rectangle(x0, y0, x1, base_y, fill=color, outline="")
        progress = min(1.0, self.index / max(1, len(self.frames)))
        self.canvas.create_rectangle(MARGIN, WINDOW_H - 11, MARGIN + usable_w * progress, WINDOW_H - 8, fill="#38bdf8", outline="")
        self.index += 1
        if self.index <= len(self.frames) + FPS * 2:
            self.root.after(int(1000 / FPS), self.draw)
        else:
            self.root.destroy()

    def run(self) -> None:
        self.root.after(40, self.draw)
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--provider", default="")
    args = parser.parse_args()
    audio = Path(args.audio)
    if not audio.exists():
        return 2
    Visualizer(audio, args.text, args.provider).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
