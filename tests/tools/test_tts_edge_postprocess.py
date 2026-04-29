"""Tests for Edge TTS Telegram/FFmpeg postprocess behavior."""

import json


def test_edge_requested_ogg_synthesizes_mp3_then_converts_to_telegram_opus(monkeypatch, tmp_path):
    from tools import tts_tool as _tt

    requested_ogg = tmp_path / "out.ogg"
    generated = {}

    async def fake_edge(text, output_path, config):
        generated["edge_output_path"] = output_path
        assert output_path.endswith(".mp3")
        with open(output_path, "wb") as f:
            f.write(b"ID3 fake mp3")
        return output_path

    def fake_postprocess(path, config):
        generated["postprocess_input"] = path
        processed = tmp_path / "out_cyber_autotune.mp3"
        processed.write_bytes(b"ID3 processed mp3")
        return str(processed)

    def fake_convert(path):
        generated["convert_input"] = path
        opus = tmp_path / "out_cyber_autotune.ogg"
        opus.write_bytes(b"OggS opus")
        return str(opus)

    monkeypatch.setattr(
        _tt,
        "_load_tts_config",
        lambda: {"provider": "edge", "postprocess": {"enabled": True, "preset": "cyber_autotune"}},
    )
    monkeypatch.setattr(_tt, "_generate_edge_tts", fake_edge)
    monkeypatch.setattr(_tt, "_apply_tts_postprocess", fake_postprocess)
    monkeypatch.setattr(_tt, "_convert_to_opus", fake_convert)
    monkeypatch.setattr(_tt, "_import_edge_tts", lambda: object())

    result = json.loads(_tt.text_to_speech_tool("안녕하세요", str(requested_ogg)))

    assert result["success"] is True
    assert result["provider"] == "edge"
    assert result["file_path"].endswith("out_cyber_autotune.ogg")
    assert result["voice_compatible"] is True
    assert result["media_tag"].startswith("[[audio_as_voice]]")
    assert generated["edge_output_path"].endswith("out.mp3")
    assert generated["postprocess_input"] == generated["edge_output_path"]
    assert generated["convert_input"].endswith("out_cyber_autotune.mp3")
