"""Tests for the optional TTS visualizer hook."""

import json


def test_tts_visualizer_hook_runs_after_success(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested = tmp_path / "out.mp3"
    called = {}

    async def fake_edge(text, output_path, config):
        with open(output_path, "wb") as f:
            f.write(b"ID3 fake mp3")
        return output_path

    def fake_visualizer(audio_path, text, provider, config):
        called["audio_path"] = audio_path
        called["text"] = text
        called["provider"] = provider
        called["config"] = config
        return True

    monkeypatch.setattr(
        _tt,
        "_load_tts_config",
        lambda: {"provider": "edge", "visualizer": {"enabled": True, "command": ["true"]}},
    )
    monkeypatch.setattr(_tt, "_generate_edge_tts", fake_edge)
    monkeypatch.setattr(_tt, "_apply_tts_postprocess", lambda path, config: None)
    monkeypatch.setattr(_tt, "_convert_to_opus", lambda path: None)
    monkeypatch.setattr(_tt, "_import_edge_tts", lambda: object())
    monkeypatch.setattr(_tt, "_notify_tts_visualizer", fake_visualizer)

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested)))

    assert result["success"] is True
    assert result["visualizer_notified"] is True
    assert called["audio_path"] == str(requested)
    assert called["text"] == "안녕하세요"
    assert called["provider"] == "edge"


def test_tts_visualizer_disabled_is_noop():
    from tools import tts_tool as _tt

    assert _tt._notify_tts_visualizer("/tmp/nope.ogg", "hello", "edge", {"visualizer": {"enabled": False}}) is False
