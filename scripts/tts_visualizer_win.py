#!/usr/bin/env python3
"""Circular Hermes TTS audio visualizer for Windows.

Resident listener launches this script in the interactive desktop session.
No third-party Python packages are required. ffmpeg/ffplay are used when present
for OGG/Opus decoding and desktop playback.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import threading
import traceback
import time
import wave
from pathlib import Path
from tkinter import BOTH, Canvas, Tk

SAMPLE_RATE = 48_000
FPS = 30
BAR_COUNT = 128
WINDOW_W = 520
WINDOW_H = 520
MINI_W = 168
MINI_H = 34
MARGIN = 18
TARGET_ALPHA = 0.92
FADE_IN_SECONDS = 0.35
FADE_OUT_SECONDS = 0.65
VISUAL_FADE_IN_SECONDS = 0.85
VISUAL_FADE_OUT_SECONDS = 0.95
HOLD_SECONDS = 1.4
HEADER_H = 34

DEFAULT_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "HermesTTSVisualizer"
SETTINGS_PATH = DEFAULT_BASE / "visualizer_settings.json"
LOG_PATH = DEFAULT_BASE / "listener.log"
PROCESS_SUSPEND_RESUME = 0x0800


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
        frames = []
        for i in range(FPS * 3):
            frames.append([0.18 + 0.12 * math.sin(i / 3 + b / 5) for b in range(BAR_COUNT)])
        return frames, 3.0

    samples_per_frame = max(1, int(sample_rate / FPS))
    frames: list[list[float]] = []
    prev = [0.0] * BAR_COUNT
    for start in range(0, len(samples), samples_per_frame):
        chunk = samples[start : start + samples_per_frame]
        if not chunk:
            continue
        band_size = max(1, len(chunk) // BAR_COUNT)
        bars: list[float] = []
        for b in range(BAR_COUNT):
            band = chunk[b * band_size : (b + 1) * band_size]
            if not band:
                bars.append(prev[b] * 0.84)
                continue
            rms = math.sqrt(sum(x * x for x in band) / len(band)) / 32768.0
            value = min(1.0, rms * 5.4)
            # Smooth enough to feel deliberate, not like a nervous ECG.
            bars.append(prev[b] * 0.58 + value * 0.42)
        prev = bars
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
        self.audio_paused = False
        self.audio_skipped = False
        self.minimized = False
        self.drag_offset: tuple[int, int] | None = None
        self.settings = load_settings()
        self.topmost = bool(self.settings.get("topmost", True))
        self.alpha_target = float(self.settings.get("alpha", TARGET_ALPHA) or TARGET_ALPHA)
        self.alpha_target = max(0.35, min(1.0, self.alpha_target))
        self.phase = random.random() * math.tau
        self.visual_fade = 0.0
        self.particles: list[dict[str, float]] = []
        self.last_particle_t = time.monotonic()

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
            x = max(0, screen_w - WINDOW_W - 24)
            y = max(0, screen_h - WINDOW_H - 70)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}+{x}+{y}")

        self.canvas = Canvas(self.root, width=WINDOW_W, height=WINDOW_H, bg="#030712", bd=0, highlightthickness=0)
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

    def _ntdll_call(self, func_name: str, pid: int) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
            handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, int(pid))
            if not handle:
                return False
            try:
                result = getattr(ntdll, func_name)(handle)
                return int(result) == 0
            finally:
                kernel32.CloseHandle(handle)
        except Exception as exc:
            log(f"{func_name} failed for pid={pid}: {exc!r}")
            return False

    def _toggle_pause_audio(self) -> None:
        proc = self.audio_proc
        if not proc or proc.poll() is not None or self.audio_skipped:
            log(f"audio pause ignored for {self.audio.name}: no active playback")
            return
        if self.audio_paused:
            if self._ntdll_call("NtResumeProcess", proc.pid):
                self.audio_paused = False
                log(f"resumed desktop audio playback for {self.audio.name} pid={proc.pid}")
            return
        if self._ntdll_call("NtSuspendProcess", proc.pid):
            self.audio_paused = True
            log(f"paused desktop audio playback for {self.audio.name} pid={proc.pid}")

    def _skip_audio(self) -> None:
        proc = self.audio_proc
        self.audio_skipped = True
        self.audio_paused = False
        if not proc or proc.poll() is not None:
            log(f"audio skip ignored for {self.audio.name}: no active playback")
            return
        try:
            proc.terminate()
            log(f"skipped desktop audio playback for {self.audio.name} pid={proc.pid}")
        except Exception as exc:
            log(f"audio skip failed for {self.audio.name}: {exc!r}")

    def _stop_playback(self) -> None:
        proc = self.audio_proc
        if not proc or proc.poll() is not None:
            return
        if self.audio_paused:
            self._ntdll_call("NtResumeProcess", proc.pid)
            self.audio_paused = False
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
        self.settings["mode"] = "kuhung_3d"
        save_settings(self.settings)

    def _button_at(self, x: int, y: int) -> str | None:
        if y > HEADER_H:
            return None
        if WINDOW_W - 35 <= x <= WINDOW_W - 12:
            return "close"
        if WINDOW_W - 64 <= x <= WINDOW_W - 41:
            return "layer"
        if WINDOW_W - 93 <= x <= WINDOW_W - 70:
            return "minimize"
        if WINDOW_W - 122 <= x <= WINDOW_W - 100:
            return "skip_audio"
        if WINDOW_W - 151 <= x <= WINDOW_W - 129:
            return "pause_audio"
        if self.minimized and 0 <= x <= MINI_W and 0 <= y <= MINI_H:
            return "restore"
        return None

    def _on_press(self, event) -> None:
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
        if button == "pause_audio":
            self._toggle_pause_audio()
            self.draw()
            return
        if button == "skip_audio":
            self._skip_audio()
            self.draw()
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

    def _smoothstep(self, value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def _visual_visibility(self, elapsed: float, total_life: float) -> float:
        # Window alpha fades the whole HUD. This second envelope fades the sphere
        # itself as a soft cyan/violet gradient so it blooms in and dissolves out,
        # instead of popping like a bureaucratic PowerPoint transition.
        in_v = self._smoothstep(elapsed / max(0.01, VISUAL_FADE_IN_SECONDS))
        remaining = total_life - elapsed
        out_v = self._smoothstep(remaining / max(0.01, VISUAL_FADE_OUT_SECONDS))
        return max(0.0, min(1.0, in_v, out_v))

    def _blend_hex(self, a: str, b: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
        br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
        return f"#{int(ar + (br - ar) * t):02x}{int(ag + (bg - ag) * t):02x}{int(ab + (bb - ab) * t):02x}"

    def _fade_hex(self, color: str, visibility: float, background: str = "#020617") -> str:
        return self._blend_hex(background, color, self._smoothstep(visibility))

    def _draw_controls(self) -> None:
        fill = "#111827"
        outline = "#334155"
        controls = [
            (WINDOW_W - 150, WINDOW_W - 129, "R" if self.audio_paused else "P", "pause_audio"),
            (WINDOW_W - 121, WINDOW_W - 100, "S", "skip_audio"),
            (WINDOW_W - 92, WINDOW_W - 70, "–", "minimize"),
            (WINDOW_W - 63, WINDOW_W - 41, "T" if self.topmost else "L", "layer"),
            (WINDOW_W - 34, WINDOW_W - 12, "×", "close"),
        ]
        for x0, x1, label, _name in controls:
            self.canvas.create_rectangle(x0, 8, x1, 26, fill=fill, outline=outline)
            self.canvas.create_text((x0 + x1) / 2, 16, text=label, fill="#cbd5e1", font=("Segoe UI", 8, "bold"))

    def _draw_header(self) -> None:
        self.canvas.create_rectangle(0, 0, WINDOW_W, HEADER_H, fill="#06111f", outline="#132033")
        self.canvas.create_text(MARGIN, 17, anchor="w", text="HERMES TTS · 3D BLOOM", fill="#f8fafc", font=("Segoe UI", 9, "bold"))
        caption_x = MARGIN + 142
        caption_right = WINDOW_W - 160
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
        self.canvas.create_text(x, 17, anchor="w", text=caption, fill="#cbd5e1", font=("Segoe UI", 8))
        self.canvas.create_rectangle(0, 0, caption_x - 2, HEADER_H, fill="#06111f", outline="")
        self.canvas.create_rectangle(caption_right + 2, 0, WINDOW_W, HEADER_H, fill="#06111f", outline="")
        self.canvas.create_text(MARGIN, 17, anchor="w", text="HERMES TTS · 3D BLOOM", fill="#f8fafc", font=("Segoe UI", 9, "bold"))
        self._draw_controls()

    def _draw_minimized(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, MINI_W, MINI_H, fill="#030712", outline="#334155")
        cx, cy = 18, 17
        pulse = 5 + 2.5 * math.sin(time.monotonic() * 4)
        self.canvas.create_oval(cx - pulse, cy - pulse, cx + pulse, cy + pulse, outline="#22d3ee", width=2)
        label = ("PAUSED" if self.audio_paused else "HERMES · 3D BLOOM") if self.topmost else "HERMES · LAYER"
        self.canvas.create_text(34, 17, anchor="w", text=label, fill="#cbd5e1", font=("Segoe UI", 8, "bold"))

    def _wait_then_start(self) -> None:
        if not self.loaded and time.monotonic() - self.load_started_at < 2.0:
            self.root.after(50, self._wait_then_start)
            return
        self.started_at = time.monotonic()
        self.index = 0
        self._start_playback()
        self.draw()

    def _current_frame(self) -> list[float]:
        return self.frames[min(self.index, len(self.frames) - 1)]

    def _frame_energy(self, frame: list[float]) -> float:
        if not frame:
            return 0.0
        return min(1.0, sum(frame) / len(frame) * 1.75)

    def _split_energy(self, frame: list[float]) -> tuple[float, float, float]:
        if not frame:
            return 0.0, 0.0, 0.0
        n = len(frame)
        low = sum(frame[: max(1, n // 4)]) / max(1, n // 4)
        mid = sum(frame[n // 4 : max(n // 4 + 1, n * 2 // 3)]) / max(1, n * 2 // 3 - n // 4)
        high = sum(frame[n * 2 // 3 :]) / max(1, n - n * 2 // 3)
        return min(1.0, low * 1.9), min(1.0, mid * 2.2), min(1.0, high * 2.8)

    def _pseudo_noise(self, x: float, y: float, z: float, t: float) -> float:
        # Lightweight Perlin-ish layered sine noise. It mirrors the WebGL shader's
        # role without pulling in a shader runtime or extra dependencies.
        return (
            math.sin(x * 1.85 + t * 1.27 + self.phase)
            + 0.62 * math.sin(y * 2.35 - t * 1.71)
            + 0.45 * math.sin((x + z) * 2.8 + t * 0.84)
            + 0.32 * math.sin(math.sqrt(x * x + y * y + z * z) * 4.2 - t * 2.1)
        ) / 2.39

    def _rotate_project(self, x: float, y: float, z: float, t: float, cx: float, cy: float, scale: float) -> tuple[float, float, float]:
        ay = t * 0.34 + self.phase * 0.25
        ax = -0.62 + math.sin(t * 0.23) * 0.12
        ca, sa = math.cos(ay), math.sin(ay)
        x, z = x * ca + z * sa, -x * sa + z * ca
        ca, sa = math.cos(ax), math.sin(ax)
        y, z = y * ca - z * sa, y * sa + z * ca
        perspective = 1.0 / max(0.34, 1.95 - z * 0.38)
        return cx + x * scale * perspective, cy + y * scale * perspective, perspective

    def _draw_glow_line(self, x0: float, y0: float, x1: float, y1: float, color: str, width: int = 1, visibility: float = 1.0) -> None:
        if visibility <= 0.015:
            return
        v = self._smoothstep(visibility)
        glow_w = max(1, int((width + 5) * (0.35 + 0.65 * v)))
        mid_w = max(1, int((width + 2) * (0.45 + 0.55 * v)))
        core_w = max(1, int(width * (0.65 + 0.35 * v)))
        self.canvas.create_line(x0, y0, x1, y1, fill=self._fade_hex("#172554", visibility), width=glow_w, capstyle="round")
        self.canvas.create_line(x0, y0, x1, y1, fill=self._fade_hex("#0e7490", visibility), width=mid_w, capstyle="round")
        self.canvas.create_line(x0, y0, x1, y1, fill=self._fade_hex(color, visibility), width=core_w, capstyle="round")

    def _draw_wire_sphere(self, cx: float, cy: float, frame: list[float], energy: float, progress: float, t: float, visibility: float = 1.0) -> None:
        lat_steps = 9
        lon_steps = 22
        v = self._smoothstep(visibility)
        sphere_scale = 0.58 + 0.42 * v
        base = 1.0 + energy * 0.12
        amp = (0.18 + energy * 0.42) * (0.35 + 0.65 * v)
        points: list[list[tuple[float, float, float]]] = []
        for lat in range(1, lat_steps):
            theta = -math.pi / 2 + math.pi * lat / lat_steps
            row = []
            for lon in range(lon_steps):
                phi = math.tau * lon / lon_steps
                sx = math.cos(theta) * math.cos(phi)
                sy = math.sin(theta)
                sz = math.cos(theta) * math.sin(phi)
                band = frame[(lon * len(frame) // lon_steps) % len(frame)] if frame else 0.0
                noise = self._pseudo_noise(sx, sy, sz, t)
                r = base + noise * amp + min(1.0, band * 2.4) * 0.16
                row.append(self._rotate_project(sx * r, sy * r, sz * r, t, cx, cy, 116 * sphere_scale))
            points.append(row)

        # Latitude rings.
        for row_i, row in enumerate(points):
            for lon in range(lon_steps):
                x0, y0, p0 = row[lon]
                x1, y1, p1 = row[(lon + 1) % lon_steps]
                depth = (p0 + p1) * 0.5
                color = "#f8fafc" if depth > 0.72 and row_i % 3 == 0 else ("#67e8f9" if row_i % 2 else "#a78bfa")
                self._draw_glow_line(x0, y0, x1, y1, color, 1 if depth < 0.72 else 2, visibility)

        # Longitude arcs.
        for lon in range(0, lon_steps, 2):
            for row_i in range(len(points) - 1):
                x0, y0, p0 = points[row_i][lon]
                x1, y1, p1 = points[row_i + 1][lon]
                depth = (p0 + p1) * 0.5
                color = "#38bdf8" if lon % 4 else "#c084fc"
                self._draw_glow_line(x0, y0, x1, y1, color, 1 if depth < 0.78 else 2, visibility)

        # Icosahedron-like triangular chords over the sphere, giving the original repo's wireframe mesh feeling.
        for lon in range(0, lon_steps, 3):
            for row_i in range(0, len(points) - 2, 2):
                a = points[row_i][lon]
                b = points[row_i + 1][(lon + 2) % lon_steps]
                c = points[row_i + 2][(lon + 1) % lon_steps]
                color = "#f0f9ff" if (lon + row_i) % 4 == 0 else "#22d3ee"
                self._draw_glow_line(a[0], a[1], b[0], b[1], color, 1, visibility)
                self._draw_glow_line(b[0], b[1], c[0], c[1], color, 1, visibility)

        # Progress halo.
        ring_r = (190 + energy * 20) * sphere_scale
        self.canvas.create_arc(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r, start=90, extent=-359.9, outline=self._fade_hex("#111827", visibility), width=5, style="arc")
        self.canvas.create_arc(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r, start=90, extent=-359.9 * progress, outline=self._fade_hex("#f8fafc", visibility), width=2, style="arc")
        self.canvas.create_arc(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r, start=90, extent=-359.9 * progress, outline=self._fade_hex("#22d3ee", visibility), width=5, style="arc")

    def _spawn_particles(self, energy: float, low: float) -> None:
        spawn = 2 + int(energy * 8) + int(low * 6)
        max_particles = 260
        for _ in range(spawn):
            if len(self.particles) >= max_particles:
                self.particles.pop(0)
            angle = random.random() * math.tau
            self.particles.append({
                "a": angle,
                "r": random.random() * 9.0,
                "vr": 1.3 + random.random() * 2.2 + energy * 4.6,
                "y": 0.0,
                "vy": 0.9 + low * 4.2 + random.random() * 1.8,
                "life": 1.0,
                "size": 1.1 + random.random() * 2.2 + energy * 2.2,
            })

    def _draw_particles(self, cx: float, cy: float, energy: float, low: float, t: float, visibility: float = 1.0) -> None:
        now = time.monotonic()
        dt = max(0.01, min(0.08, now - self.last_particle_t))
        self.last_particle_t = now
        if energy > 0.035 and visibility > 0.18:
            self._spawn_particles(energy * visibility, low * visibility)
        alive: list[dict[str, float]] = []
        for p in self.particles:
            p["r"] += p["vr"] * dt * 42
            p["vy"] -= 4.9 * dt
            p["y"] = max(0.0, p["y"] + p["vy"] * dt * 34)
            p["life"] -= dt * (0.42 + p["r"] / 620)
            if p["life"] <= 0 or p["r"] > 230:
                continue
            alive.append(p)
            a = p["a"] + math.sin(t * 0.7 + p["r"] * 0.01) * 0.12
            x = cx + math.cos(a) * p["r"]
            z = math.sin(a) * p["r"]
            y = cy + z * 0.34 - p["y"]
            fade = max(0.0, min(1.0, p["life"]))
            particle_visibility = visibility * fade
            size = p["size"] * (0.45 + fade) * (0.45 + 0.55 * self._smoothstep(visibility))
            color = "#f8fafc" if p["r"] < 44 else ("#67e8f9" if int(p["r"]) % 2 else "#a78bfa")
            self.canvas.create_oval(x - size * 2.4, y - size * 2.4, x + size * 2.4, y + size * 2.4, outline=self._fade_hex("#0e7490", particle_visibility), width=1)
            self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=self._fade_hex(color, particle_visibility), outline="")
        self.particles = alive

    def _draw_circular_body(self, frame: list[float], progress: float, visibility: float = 1.0) -> None:
        cx = WINDOW_W / 2
        cy = HEADER_H + (WINDOW_H - HEADER_H) / 2 + 14
        energy = self._frame_energy(frame)
        low, mid, high = self._split_energy(frame)
        t = time.monotonic() - self.started_at

        # Dark WebGL-like scene background with bloom-friendly rings.
        self.canvas.create_rectangle(0, HEADER_H, WINDOW_W, WINDOW_H, fill="#020617", outline="")
        v = self._smoothstep(visibility)
        scene_scale = 0.82 + 0.18 * v
        for k, (r, color) in enumerate([(238, "#050b1e"), (205, "#081427"), (166, "#0b1735"), (126, "#061d2d")]):
            rr = r * scene_scale
            self.canvas.create_oval(cx - rr, cy - rr * 0.82, cx + rr, cy + rr * 0.82, outline=self._fade_hex(color, visibility), width=max(1, int((8 - k) * (0.55 + 0.45 * v))))
        for r, color, width in [(218, "#172554", 1), (174, "#0e7490", 1), (132, "#312e81", 1), (84, "#164e63", 1)]:
            rr = r * scene_scale
            self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=self._fade_hex(color, visibility), width=width)

        # Particle/ripple system: center spawn, outward spread, vertical kick.
        self._draw_particles(cx, cy + 42, energy, low, t, visibility)

        # Shader-like displaced wireframe 3D object.
        self._draw_wire_sphere(cx, cy - 8, frame, energy, progress, t, visibility)

        # Core label and status.
        core = (42 + low * 16 + mid * 10) * (0.72 + 0.28 * v)
        self.canvas.create_oval(cx - core * 1.55, cy - 8 - core * 1.55, cx + core * 1.55, cy - 8 + core * 1.55, outline=self._fade_hex("#0e7490", visibility), width=max(1, int(7 * (0.45 + 0.55 * v))))
        self.canvas.create_oval(cx - core, cy - 8 - core, cx + core, cy - 8 + core, fill="#020617", outline=self._fade_hex("#f8fafc", visibility), width=2)
        self.canvas.create_text(cx, cy - 22, text="HERMES", fill=self._fade_hex("#f8fafc", visibility), font=("Segoe UI", 17, "bold"))
        status = "SKIPPED" if self.audio_skipped else ("PAUSED" if self.audio_paused else (self.provider or "3D AUDIO").upper()[:18])
        self.canvas.create_text(cx, cy + 5, text=status, fill=self._fade_hex("#67e8f9", visibility), font=("Segoe UI", 8, "bold"))
        self.canvas.create_text(cx, cy + 22, text="WIREFRAME · BLOOM · PARTICLES", fill=self._fade_hex("#a5b4fc", visibility), font=("Segoe UI", 7, "bold"))

    def draw(self) -> None:
        elapsed = time.monotonic() - self.started_at
        total_life = max(self.duration, len(self.frames) / FPS) + HOLD_SECONDS + FADE_OUT_SECONDS
        if not self._apply_timed_alpha(elapsed, total_life):
            self._close_window()
            return
        visibility = self._visual_visibility(elapsed, total_life)
        self.visual_fade = visibility
        if self.minimized:
            self._draw_minimized()
            self.root.after(int(1000 / FPS), self.draw)
            return

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, WINDOW_W, WINDOW_H, fill="#030712", outline="#172033")
        self._draw_header()
        frame = self._current_frame()
        progress = min(1.0, self.index / max(1, len(self.frames)))
        self._draw_circular_body(frame, progress, visibility)
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
    try:
        raise SystemExit(main())
    except Exception:
        log("visualizer fatal error:\n" + traceback.format_exc())
        raise
