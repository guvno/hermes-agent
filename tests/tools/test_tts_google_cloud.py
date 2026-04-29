"""Tests for the Google Cloud Text-to-Speech provider."""

import json
from types import SimpleNamespace
from unittest.mock import patch


def test_google_cloud_tts_uses_chirp3_hd_defaults(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    captured = {}

    class FakeTextToSpeechClient:
        def synthesize_speech(self, request):
            captured["request"] = request
            return SimpleNamespace(audio_content=b"RIFF fake wav")

    class FakeTextToSpeech:
        TextToSpeechClient = FakeTextToSpeechClient
        SynthesisInput = lambda self, text: {"text": text}
        VoiceSelectionParams = lambda self, language_code, name: {
            "language_code": language_code,
            "name": name,
        }
        AudioConfig = lambda self, audio_encoding, speaking_rate, pitch: {
            "audio_encoding": audio_encoding,
            "speaking_rate": speaking_rate,
            "pitch": pitch,
        }
        AudioEncoding = SimpleNamespace(LINEAR16="LINEAR16", MP3="MP3")

    monkeypatch.setattr(_tt, "_import_google_cloud_tts", lambda: FakeTextToSpeech())

    output = tmp_path / "out.wav"
    result = _tt._generate_google_cloud_tts("안녕하세요", str(output), {})

    assert result == str(output)
    assert output.read_bytes() == b"RIFF fake wav"
    request = captured["request"]
    assert request["input"] == {"text": "안녕하세요"}
    assert request["voice"]["language_code"] == "ko-KR"
    assert request["voice"]["name"] == "ko-KR-Chirp3-HD-Kore"
    assert request["audio_config"]["audio_encoding"] == "LINEAR16"
    assert request["audio_config"]["speaking_rate"] == 1.0
    assert request["audio_config"]["pitch"] == 0.0


def test_google_cloud_tts_accepts_custom_voice_and_rate(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    captured = {}

    class FakeTextToSpeechClient:
        def synthesize_speech(self, request):
            captured["request"] = request
            return SimpleNamespace(audio_content=b"mp3 bytes")

    class FakeTextToSpeech:
        TextToSpeechClient = FakeTextToSpeechClient
        SynthesisInput = lambda self, text: {"text": text}
        VoiceSelectionParams = lambda self, language_code, name: {
            "language_code": language_code,
            "name": name,
        }
        AudioConfig = lambda self, audio_encoding, speaking_rate, pitch: {
            "audio_encoding": audio_encoding,
            "speaking_rate": speaking_rate,
            "pitch": pitch,
        }
        AudioEncoding = SimpleNamespace(LINEAR16="LINEAR16", MP3="MP3")

    monkeypatch.setattr(_tt, "_import_google_cloud_tts", lambda: FakeTextToSpeech())

    config = {
        "google_cloud": {
            "language_code": "ko-KR",
            "voice": "ko-KR-Chirp3-HD-Aoede",
            "audio_encoding": "MP3",
            "speaking_rate": 1.08,
            "pitch": -1.5,
        }
    }
    output = tmp_path / "out.mp3"
    _tt._generate_google_cloud_tts("안녕하세요", str(output), config)

    request = captured["request"]
    assert request["voice"]["name"] == "ko-KR-Chirp3-HD-Aoede"
    assert request["audio_config"]["audio_encoding"] == "MP3"
    assert request["audio_config"]["speaking_rate"] == 1.08
    assert request["audio_config"]["pitch"] == -1.5
    assert output.read_bytes() == b"mp3 bytes"


def test_text_to_speech_dispatches_google_cloud_to_wav_then_telegram_opus(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested_ogg = tmp_path / "out.ogg"
    generated = {}

    def fake_google_cloud(text, output_path, config):
        generated["output_path"] = output_path
        assert output_path.endswith(".wav")
        with open(output_path, "wb") as f:
            f.write(b"RIFF fake wav")
        return output_path

    def fake_postprocess(path, config):
        generated["postprocess_input"] = path
        processed = tmp_path / "out_cyber_autotune.wav"
        processed.write_bytes(b"RIFF processed wav")
        return str(processed)

    def fake_convert(path):
        generated["convert_input"] = path
        opus = tmp_path / "out_cyber_autotune.ogg"
        opus.write_bytes(b"OggS opus")
        return str(opus)

    monkeypatch.setattr(
        _tt,
        "_load_tts_config",
        lambda: {"provider": "google_cloud", "postprocess": {"enabled": True, "preset": "cyber_autotune"}},
    )
    monkeypatch.setattr(_tt, "_generate_google_cloud_tts", fake_google_cloud)
    monkeypatch.setattr(_tt, "_apply_tts_postprocess", fake_postprocess)
    monkeypatch.setattr(_tt, "_convert_to_opus", fake_convert)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested_ogg)))

    assert result["success"] is True
    assert result["provider"] == "google_cloud"
    assert result["file_path"].endswith("out_cyber_autotune.ogg")
    assert result["voice_compatible"] is True
    assert result["media_tag"].startswith("[[audio_as_voice]]")
    assert generated["output_path"].endswith("out.wav")
    assert generated["postprocess_input"] == generated["output_path"]
    assert generated["convert_input"].endswith("out_cyber_autotune.wav")


def test_text_to_speech_falls_back_to_edge_when_google_cloud_monthly_quota_exhausted(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested_ogg = tmp_path / "quota_fallback.ogg"
    generated = {}

    def fake_google_cloud(text, output_path, config):
        generated["google_cloud_called"] = True
        raise ValueError("Google Cloud TTS monthly character limit would be exceeded: used 999,999/1,000,000")

    async def fake_edge(text, output_path, config):
        generated["edge_output_path"] = output_path
        assert output_path.endswith(".mp3")
        with open(output_path, "wb") as f:
            f.write(b"fake edge mp3")
        return output_path

    def fake_convert(path):
        generated["convert_input"] = path
        opus = tmp_path / "quota_fallback.ogg"
        opus.write_bytes(b"OggS opus")
        return str(opus)

    monkeypatch.setattr(
        _tt,
        "_load_tts_config",
        lambda: {
            "provider": "google_cloud",
            "google_cloud": {"fallback_provider": "edge"},
            "edge": {"voice": "ko-KR-SunHiNeural"},
            "postprocess": {"enabled": False, "preset": "none"},
        },
    )
    monkeypatch.setattr(_tt, "_generate_google_cloud_tts", fake_google_cloud)
    monkeypatch.setattr(_tt, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(_tt, "_generate_edge_tts", fake_edge)
    monkeypatch.setattr(_tt, "_convert_to_opus", fake_convert)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested_ogg)))

    assert result["success"] is True
    assert result["provider"] == "edge"
    assert result["fallback_from"] == "google_cloud"
    assert "monthly character limit" in result["fallback_reason"]
    assert result["file_path"].endswith("quota_fallback.ogg")
    assert result["voice_compatible"] is True
    assert generated["google_cloud_called"] is True
    assert generated["edge_output_path"].endswith("quota_fallback.mp3")
    assert generated["convert_input"] == generated["edge_output_path"]


def test_google_cloud_has_default_text_limit():
    from tools.tts_tool import PROVIDER_MAX_TEXT_LENGTH, _resolve_max_text_length

    assert PROVIDER_MAX_TEXT_LENGTH["google_cloud"] == 5000
    assert _resolve_max_text_length("google_cloud", {}) == 5000


def test_google_cloud_records_successful_monthly_character_usage(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    monkeypatch.setattr(_tt, "_google_cloud_usage_month", lambda: "2026-04")
    monkeypatch.setattr(_tt, "_google_cloud_usage_path", lambda config: tmp_path / "usage.json")

    class FakeTextToSpeechClient:
        def synthesize_speech(self, request):
            return SimpleNamespace(audio_content=b"RIFF fake wav")

    class FakeTextToSpeech:
        TextToSpeechClient = FakeTextToSpeechClient
        SynthesisInput = lambda self, text: {"text": text}
        VoiceSelectionParams = lambda self, language_code, name: {
            "language_code": language_code,
            "name": name,
        }
        AudioConfig = lambda self, audio_encoding, speaking_rate, pitch: {
            "audio_encoding": audio_encoding,
            "speaking_rate": speaking_rate,
            "pitch": pitch,
        }
        AudioEncoding = SimpleNamespace(LINEAR16="LINEAR16", MP3="MP3")

    monkeypatch.setattr(_tt, "_import_google_cloud_tts", lambda: FakeTextToSpeech())

    _tt._generate_google_cloud_tts("가나다", str(tmp_path / "out.wav"), {})

    usage = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert usage == {"month": "2026-04", "characters": 3}


def test_google_cloud_monthly_character_limit_blocks_before_api_call(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    usage_path = tmp_path / "usage.json"
    usage_path.write_text(json.dumps({"month": "2026-04", "characters": 999_998}), encoding="utf-8")
    monkeypatch.setattr(_tt, "_google_cloud_usage_month", lambda: "2026-04")
    monkeypatch.setattr(_tt, "_google_cloud_usage_path", lambda config: usage_path)

    called = {"api": False}

    class FakeTextToSpeechClient:
        def synthesize_speech(self, request):
            called["api"] = True
            return SimpleNamespace(audio_content=b"RIFF fake wav")

    class FakeTextToSpeech:
        TextToSpeechClient = FakeTextToSpeechClient
        SynthesisInput = lambda self, text: {"text": text}
        VoiceSelectionParams = lambda self, language_code, name: {
            "language_code": language_code,
            "name": name,
        }
        AudioConfig = lambda self, audio_encoding, speaking_rate, pitch: {
            "audio_encoding": audio_encoding,
            "speaking_rate": speaking_rate,
            "pitch": pitch,
        }
        AudioEncoding = SimpleNamespace(LINEAR16="LINEAR16", MP3="MP3")

    monkeypatch.setattr(_tt, "_import_google_cloud_tts", lambda: FakeTextToSpeech())

    try:
        _tt._generate_google_cloud_tts("초과됨", str(tmp_path / "out.wav"), {})
    except ValueError as e:
        assert "monthly character limit" in str(e)
        assert "1,000,000" in str(e)
    else:
        raise AssertionError("Expected monthly quota ValueError")

    assert called["api"] is False
    assert json.loads(usage_path.read_text(encoding="utf-8"))["characters"] == 999_998


def test_google_cloud_monthly_character_limit_can_be_configured(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    usage_path = tmp_path / "usage.json"
    usage_path.write_text(json.dumps({"month": "2026-04", "characters": 10}), encoding="utf-8")
    monkeypatch.setattr(_tt, "_google_cloud_usage_month", lambda: "2026-04")
    monkeypatch.setattr(_tt, "_google_cloud_usage_path", lambda config: usage_path)

    class FakeTextToSpeechClient:
        def synthesize_speech(self, request):
            return SimpleNamespace(audio_content=b"RIFF fake wav")

    class FakeTextToSpeech:
        TextToSpeechClient = FakeTextToSpeechClient
        SynthesisInput = lambda self, text: {"text": text}
        VoiceSelectionParams = lambda self, language_code, name: {
            "language_code": language_code,
            "name": name,
        }
        AudioConfig = lambda self, audio_encoding, speaking_rate, pitch: {
            "audio_encoding": audio_encoding,
            "speaking_rate": speaking_rate,
            "pitch": pitch,
        }
        AudioEncoding = SimpleNamespace(LINEAR16="LINEAR16", MP3="MP3")

    monkeypatch.setattr(_tt, "_import_google_cloud_tts", lambda: FakeTextToSpeech())
    config = {"google_cloud": {"monthly_character_limit": 20}}

    _tt._generate_google_cloud_tts("12345", str(tmp_path / "ok.wav"), config)
    assert json.loads(usage_path.read_text(encoding="utf-8"))["characters"] == 15

    try:
        _tt._generate_google_cloud_tts("123456", str(tmp_path / "blocked.wav"), config)
    except ValueError as e:
        assert "20" in str(e)
    else:
        raise AssertionError("Expected configured monthly quota ValueError")
