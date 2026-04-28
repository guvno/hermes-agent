"""Subprocess runner for MeloTTS synthesis.

This module is intentionally tiny so Hermes can call a separate MeloTTS virtualenv
without importing MeloTTS/Torch into the main Hermes process.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate speech with MeloTTS")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="KR")
    parser.add_argument("--speaker", default="KR")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from melo.api import TTS

    model = TTS(language=args.language, device=args.device)
    speaker_ids = model.hps.data.spk2id
    speaker = args.speaker or args.language
    try:
        speaker_id = speaker_ids[speaker]
    except Exception as exc:
        try:
            available = ", ".join(sorted(speaker_ids.keys()))
        except Exception:
            available = str(speaker_ids)
        raise SystemExit(f"Unknown MeloTTS speaker {speaker!r}; available: {available}") from exc

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    model.tts_to_file(args.text, speaker_id, str(output), speed=args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
