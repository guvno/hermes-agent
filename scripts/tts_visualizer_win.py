#!/usr/bin/env python3
"""Always-on-top corner visualizer for Hermes TTS audio on Windows.

No third-party Python packages are required. ffmpeg is used when available so
Telegram-friendly OGG/Opus files work; WAV files fall back to the stdlib wave
module.
"""
from __future__ import annotations

import argparse
import json
import math
import os
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
MINI_W = 168
MINI_H = 34
MARGIN = 16
TARGET_ALPHA = 0.92
FADE_IN_SECONDS = 0.35
FADE_OUT_SECONDS = 0.65
HOLD_SECONDS = 1.4
HEADER_H = 31

DEFAULT_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "HermesTTSVisualizer"
SETTINGS_PATH = DEFAULT_BASE / "visualizer_settings.json"
LOG_PATH = DEFAULT_BASE / "listener.log"


def log(message: str) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except Exception:
        pass


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


def load_settings() -> dict[str, object]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data: dict[str, object]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


class Visualizer:
    def __init__(self, audio: Path, text: str, provider: str, play_audio: bool = True):
        self.audio = audio
        self.text = " ".join((text or audio.name).replace("\n", " ").split())
        self.provider = provider
        self.frames: list[list[float]] = [[0.0] * BAR_COUNT]
        self.duration = 2.0
        self.index = 0
        self.started_at = time.monotonic()
        self.loaded = False
        self.load_started_at = time.monotonic()
        self.play_audio = play_audio
        self.play_started = False
        self.audio_proc: subprocess.Popen | None = None
        self.minimized = False
        self.drag_offset: tuple[int, int] | None = None
        self.settings = load_settings()
        self.topmost = bool(self.settings.get("topmost", True))
        self.alpha_target = float(self.settings.get("alpha", TARGET_ALPHA) or TARGET_ALPHA)
        self.alpha_target = max(0.35, min(1.0, self.alpha_target))

        self.root = Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", self.topmost)
        try:
            self.root.attributes("-toolwindow", True)
        except Exception:
            pass
        self._set_alpha(0.0)

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        saved_x = self.settings.get("x")
        saved_y = self.settings.get("y")
        if isinstance(saved_x, int) and isinstance(saved_y, int):
            x = clamp(saved_x, 0, max(0, screen_w - WINDOW_W))
            y = clamp(saved_y, 0, max(0, screen_h - WINDOW_H))
        else:
            x = max(0, screen_w - WINDOW_W - 18)
            y = max(0, screen_h - WINDOW_H - 58)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")

        self.canvas = Canvas(self.root, width=WINDOW_W, height=WINDOW_H, bg="#05070d", bd=0, highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.toggle_minimize())
        self.root.bind("<Escape>", lambda _event: self.fade_close())
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        self.frames, self.duration = load_envelope(self.audio)
        self.loaded = True

    def _find_ffplay(self) -> str | None:
        ffplay = shutil.which("ffplay")
        if ffplay:
            return ffplay
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            candidate = Path(ffmpeg).with_name("ffplay.exe" if sys.platform.startswith("win") else "ffplay")
            if candidate.exists():
                return str(candidate)
        return None

    def _start_playback(self) -> None:
        if self.play_started or not self.play_audio:
            return
        self.play_started = True
        ffplay = self._find_ffplay()
        if not ffplay:
            log(f"audio playback skipped for {self.audio.name}: ffplay not found")
            return
        cmd = [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(self.audio)]
        try:
            creationflags = 0
            if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags = subprocess.CREATE_NO_WINDOW
            self.audio_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            log(f"started desktop audio playback for {self.audio.name} pid={self.audio_proc.pid}")
        except Exception as exc:
            log(f"audio playback failed for {self.audio.name}: {exc!r}")
            self.audio_proc = None

    def _stop_playback(self) -> None:
        proc = self.audio_proc
        if not proc or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            pass

    def _close_window(self) -> None:
        self._stop_playback()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _set_alpha(self, alpha: float) -> None:
        try:
            self.root.attributes("-alpha", max(0.0, min(1.0, alpha)))
        except Exception:
            pass

    def _window_pos(self) -> tuple[int, int]:
        self.root.update_idletasks()
        return self.root.winfo_x(), self.root.winfo_y()

    def _save_position(self) -> None:
        x, y = self._window_pos()
        self.settings["x"] = int(x)
        self.settings["y"] = int(y)
        self.settings["topmost"] = bool(self.topmost)
        self.settings["alpha"] = float(self.alpha_target)
        save_settings(self.settings)

    def _button_at(self, x: int, y: int) -> str | None:
        if y > HEADER_H:
            return None
        # Right-side custom controls: minimize, layer/topmost, close.
        if WINDOW_W - 35 <= x <= WINDOW_W - 12:
            return "close"
        if WINDOW_W - 64 <= x <= WINDOW_W - 41:
            return "layer"
        if WINDOW_W - 93 <= x <= WINDOW_W - 70:
            return "minimize"
        if self.minimized and 0 <= x <= MINI_W and 0 <= y <= MINI_H:
            return "restore"
        return None

    def _on_press(self, event) -> None:  # tkinter event, intentionally untyped
        button = self._button_at(event.x, event.y)
        if button == "close":
            self.fade_close()
            return
        if button == "layer":
            self.toggle_layer()
            return
        if button in {"minimize", "restore"}:
            self.toggle_minimize()
            return
        self.drag_offset = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _on_drag(self, event) -> None:
        if not self.drag_offset:
            return
        dx, dy = self.drag_offset
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")

    def _on_release(self, _event) -> None:
        if self.drag_offset:
            self.drag_offset = None
            self._save_position()

    def toggle_minimize(self) -> None:
        self.minimized = not self.minimized
        x, y = self._window_pos()
        if self.minimized:
            self.root.geometry(f"{MINI_W}x{MINI_H}+{x}+{y}")
            self.canvas.config(width=MINI_W, height=MINI_H)
        else:
            self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")
            self.canvas.config(width=WINDOW_W, height=WINDOW_H)
        self.draw()

    def toggle_layer(self) -> None:
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)
        if not self.topmost:
            try:
                self.root.lower()
            except Exception:
                pass
        self._save_position()
        self.draw()

    def fade_close(self) -> None:
        self.index = len(self.frames) + int(HOLD_SECONDS * FPS) + 1
        self._fade_out_step()

    def _fade_out_step(self) -> None:
        try:
            alpha = float(self.root.attributes("-alpha"))
        except Exception:
            alpha = self.alpha_target
        next_alpha = alpha - max(0.03, self.alpha_target / (FADE_OUT_SECONDS * FPS))
        if next_alpha <= 0.02:
            self._close_window()
            return
        self._set_alpha(next_alpha)
        self.root.after(int(1000 / FPS), self._fade_out_step)

    def _apply_timed_alpha(self, elapsed: float, total_life: float) -> bool:
        if elapsed < FADE_IN_SECONDS:
            self._set_alpha(self.alpha_target * (elapsed / FADE_IN_SECONDS))
            return True
        remaining = total_life - elapsed
        if remaining < FADE_OUT_SECONDS:
            if remaining <= 0:
                self._set_alpha(0.0)
                return False
            self._set_alpha(self.alpha_target * (remaining / FADE_OUT_SECONDS))
            return True
        self._set_alpha(self.alpha_target)
        return True

    def _draw_controls(self) -> None:
        # Small, intentionally quiet controls. Apparently windows behave better when
        # given manners and a few buttons.
        fill = "#162033"
        outline = "#334155"
        controls = [
            (WINDOW_W - 92, WINDOW_W - 70, "—", "minimize"),
            (WINDOW_W - 63, WINDOW_W - 41, "T" if self.topmost else "L", "layer"),
            (WINDOW_W - 34, WINDOW_W - 12, "×", "close"),
        ]
        for x0, x1, label, _name in controls:
            self.canvas.create_rectangle(x0, 7, x1, 25, fill=fill, outline=outline)
            self.canvas.create_text((x0 + x1) / 2, 15, text=label, fill="#cbd5e1", font=("Segoe UI", 8, "bold"))

    def _draw_header(self) -> None:
        self.canvas.create_rectangle(0, 0, WINDOW_W, HEADER_H, fill="#07111f", outline="#152033")
        self.canvas.create_text(MARGIN, 15, anchor="w", text="HERMES TTS", fill="#7dd3fc", font=("Segoe UI", 9, "bold"))
        caption_x = MARGIN + 92
        caption_right = WINDOW_W - 104
        caption_w = max(40, caption_right - caption_x)
        caption = self.text or self.audio.name
        item = self.canvas.create_text(0, -100, anchor="w", text=caption, fill="#cbd5e1", font=("Segoe UI", 8))
        bbox = self.canvas.bbox(item)
        text_w = (bbox[2] - bbox[0]) if bbox else caption_w
        self.canvas.delete(item)
        if text_w <= caption_w:
            x = caption_x
        else:
            cycle_w = text_w + caption_w + 42
            offset = (time.monotonic() - self.started_at) * 42 % cycle_w
            x = caption_x + caption_w - offset
        self.canvas.create_text(x, 15, anchor="w", text=caption, fill="#cbd5e1", font=("Segoe UI", 8))
        # Mask the marquee so it does not wander into title/buttons like a drunk ticker.
        self.canvas.create_rectangle(0, 0, caption_x - 2, HEADER_H, fill="#07111f", outline="")
        self.canvas.create_rectangle(caption_right + 2, 0, WINDOW_W, HEADER_H, fill="#07111f", outline="")
        self.canvas.create_text(MARGIN, 15, anchor="w", text="HERMES TTS", fill="#7dd3fc", font=("Segoe UI", 9, "bold"))
        self._draw_controls()

    def _draw_minimized(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, MINI_W, MINI_H, fill="#05070d", outline="#334155")
        self.canvas.create_oval(10, 11, 22, 23, fill="#22d3ee", outline="")
        label = "HERMES TTS  ▣" if self.topmost else "HERMES TTS  □"
        self.canvas.create_text(31, 17, anchor="w", text=label, fill="#cbd5e1", font=("Segoe UI", 8, "bold"))

    def _wait_then_start(self) -> None:
        # Keep playback and bars aligned: wait briefly for ffmpeg decoding before
        # starting ffplay. If decoding is slow or unavailable, start anyway rather
        # than leaving a polite but useless dark rectangle.
        if not self.loaded and time.monotonic() - self.load_started_at < 2.0:
            self.root.after(50, self._wait_then_start)
            return
        self.started_at = time.monotonic()
        self.index = 0
        self._start_playback()
        self.draw()

    def draw(self) -> None:
        elapsed = time.monotonic() - self.started_at
        total_life = max(self.duration, len(self.frames) / FPS) + HOLD_SECONDS + FADE_OUT_SECONDS
        if not self._apply_timed_alpha(elapsed, total_life):
            self._close_window()
            return
        if self.minimized:
            self._draw_minimized()
            self.root.after(int(1000 / FPS), self.draw)
            return

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, WINDOW_W, WINDOW_H, fill="#05070d", outline="#152033")
        self._draw_header()
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        usable_w = WINDOW_W - MARGIN * 2
        bar_w = usable_w / BAR_COUNT
        base_y = WINDOW_H - 25
        for i, amp in enumerate(frame):
            eased = min(1.0, amp ** 0.55)
            h = 7 + eased * 82
            x0 = MARGIN + i * bar_w + 1
            x1 = MARGIN + (i + 1) * bar_w - 1
            y0 = base_y - h
            color = "#22d3ee" if i % 3 else "#a78bfa"
            self.canvas.create_rectangle(x0, y0, x1, base_y, fill=color, outline="")
        progress = min(1.0, self.index / max(1, len(self.frames)))
        self.canvas.create_rectangle(MARGIN, WINDOW_H - 11, MARGIN + usable_w * progress, WINDOW_H - 8, fill="#38bdf8", outline="")
        self.index += 1
        if elapsed <= total_life:
            self.root.after(int(1000 / FPS), self.draw)
        else:
            self._close_window()

    def run(self) -> None:
        self.root.after(40, self._wait_then_start)
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--play", dest="play_audio", action="store_true", default=True, help="Play the same audio on this Windows desktop while visualizing")
    parser.add_argument("--no-play", dest="play_audio", action="store_false", help="Visualize only; do not play audio on this Windows desktop")
    args = parser.parse_args()
    audio = Path(args.audio)
    if not audio.exists():
        return 2
    Visualizer(audio, args.text, args.provider, play_audio=args.play_audio).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
