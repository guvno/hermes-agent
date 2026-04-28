"""Tests for the local Piper HTTP TTS provider."""

import json
import urllib.request
from unittest.mock import MagicMock


class FakeUrlopenResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, *args):
        return self.body


def test_check_piper_available_uses_health_endpoint(monkeypatch):
    from tools.tts_tool import _check_piper_available

    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append((request.full_url, timeout))
        return FakeUrlopenResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert _check_piper_available({"piper": {"base_url": "http://127.0.0.1:5005"}}) is True
    assert calls == [("http://127.0.0.1:5005/health", 2)]


def test_generate_piper_posts_text_and_writes_wav(monkeypatch, tmp_path):
    from tools.tts_tool import _generate_piper_tts

    calls = []
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav"

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        assert timeout == 12
        return FakeUrlopenResponse(wav_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    output = tmp_path / "out.wav"
    result = _generate_piper_tts(
        "안녕하세요",
        str(output),
        {"piper": {"base_url": "http://127.0.0.1:5005", "timeout": 12}},
    )

    assert result == str(output)
    assert output.read_bytes() == wav_bytes
    assert len(calls) == 1
    request = calls[0]
    assert request.full_url == "http://127.0.0.1:5005/tts"
    assert json.loads(request.data.decode("utf-8")) == {"text": "안녕하세요"}
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"


def test_text_to_speech_dispatches_to_piper(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    captured = {}

    def fake_piper(text, output_path, config):
        captured["text"] = text
        captured["output_path"] = output_path
        captured["config"] = config
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav")
        return output_path

    monkeypatch.setattr(_tt, "_load_tts_config", lambda: {"provider": "piper"})
    monkeypatch.setattr(_tt, "_generate_piper_tts", fake_piper)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(tmp_path / "out.wav")))

    assert result["success"] is True
    assert result["provider"] == "piper"
    assert captured["text"] == "안녕하세요"


def test_piper_requested_ogg_is_converted_to_telegram_opus(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested_ogg = tmp_path / "out.ogg"
    generated = {}

    def fake_piper(text, output_path, config):
        generated["output_path"] = output_path
        with open(output_path, "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt fake-wav")
        return output_path

    def fake_convert(path):
        generated["convert_input"] = path
        requested_ogg.write_bytes(b"OggS opus")
        return str(requested_ogg)

    monkeypatch.setattr(_tt, "_load_tts_config", lambda: {"provider": "piper"})
    monkeypatch.setattr(_tt, "_generate_piper_tts", fake_piper)
    monkeypatch.setattr(_tt, "_convert_to_opus", fake_convert)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested_ogg)))

    assert result["success"] is True
    assert result["file_path"] == str(requested_ogg)
    assert result["voice_compatible"] is True
    assert result["media_tag"].startswith("[[audio_as_voice]]")
    assert generated["output_path"].endswith(".wav")
    assert generated["convert_input"] == generated["output_path"]


def test_piper_has_default_text_limit():
    from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH, _resolve_max_text_length

    assert PROVIDER_MAX_TEXT_LENGTH["piper"] == 2000
    assert _resolve_max_text_length("piper", {}) == 2000
