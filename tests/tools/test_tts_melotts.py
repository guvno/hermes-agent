"""Tests for the local MeloTTS provider."""

import json
import os
import subprocess


def test_generate_melotts_invokes_external_python(monkeypatch, tmp_path):
    from tools.tts_tool import _generate_melotts_tts

    calls = []

    def fake_run(cmd, check, timeout, env):
        calls.append((cmd, check, timeout, env))
        output_path = cmd[-1]
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = tmp_path / "out.wav"
    result = _generate_melotts_tts(
        "안녕하세요",
        str(output),
        {
            "melotts": {
                "python": "/home/ubuntu/melotts-venv/bin/python",
                "language": "KR",
                "speaker": "KR",
                "speed": 1.1,
                "device": "cpu",
                "timeout": 123,
            }
        },
    )

    assert result == str(output)
    assert output.read_bytes().startswith(b"RIFF")
    assert len(calls) == 1
    cmd, check, timeout, env = calls[0]
    assert cmd[0] == "/home/ubuntu/melotts-venv/bin/python"
    assert cmd[1:3] == ["-m", "tools.melotts_runner"]
    assert cmd[cmd.index("--language") + 1] == "KR"
    assert cmd[cmd.index("--speaker") + 1] == "KR"
    assert cmd[cmd.index("--speed") + 1] == "1.1"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--text") + 1] == "안녕하세요"
    assert cmd[-1] == str(output)
    assert check is True
    assert timeout == 123
    assert env["TOKENIZERS_PARALLELISM"] == "false"


def test_melotts_missing_output_raises(monkeypatch, tmp_path):
    from tools.tts_tool import _generate_melotts_tts

    def fake_run(cmd, check, timeout, env):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    try:
        _generate_melotts_tts("안녕하세요", str(tmp_path / "missing.wav"), {"melotts": {}})
    except FileNotFoundError as exc:
        assert "produced no output" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_text_to_speech_dispatches_to_melotts(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    captured = {}

    def fake_melotts(text, output_path, config):
        captured["text"] = text
        captured["output_path"] = output_path
        Path = __import__("pathlib").Path
        Path(output_path).write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav")
        return output_path

    monkeypatch.setattr(_tt, "_load_tts_config", lambda: {"provider": "melotts"})
    monkeypatch.setattr(_tt, "_generate_melotts_tts", fake_melotts)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(tmp_path / "out.wav")))

    assert result["success"] is True
    assert result["provider"] == "melotts"
    assert captured["text"] == "안녕하세요"


def test_melotts_requested_ogg_is_converted_to_telegram_opus(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested_ogg = tmp_path / "out.ogg"
    generated = {}

    def fake_melotts(text, output_path, config):
        generated["output_path"] = output_path
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav")
        return output_path

    def fake_convert(path):
        generated["convert_input"] = path
        requested_ogg.write_bytes(b"OggS opus")
        return str(requested_ogg)

    monkeypatch.setattr(_tt, "_load_tts_config", lambda: {"provider": "melotts"})
    monkeypatch.setattr(_tt, "_generate_melotts_tts", fake_melotts)
    monkeypatch.setattr(_tt, "_convert_to_opus", fake_convert)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested_ogg)))

    assert result["success"] is True
    assert result["file_path"] == str(requested_ogg)
    assert result["voice_compatible"] is True
    assert result["media_tag"].startswith("[[audio_as_voice]]")
    assert generated["output_path"].endswith(".wav")
    assert generated["convert_input"] == generated["output_path"]


def test_melotts_has_default_text_limit():
    from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH, _resolve_max_text_length

    assert PROVIDER_MAX_TEXT_LENGTH["melotts"] == 2000
    assert _resolve_max_text_length("melotts", {}) == 2000
