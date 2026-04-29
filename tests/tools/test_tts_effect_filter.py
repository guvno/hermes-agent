"""Tests for TTS FFmpeg effect filter construction."""


def test_cyber_autotune_pitch_chain_uses_input_sample_rate(monkeypatch):
    from tools import tts_tool as _tt

    monkeypatch.delenv("HERMES_TTS_USE_RUBBERBAND", raising=False)

    chain = _tt._tts_effect_filter("cyber_autotune", sample_rate=24000)

    assert "asetrate=24000*1.05" in chain
    assert "aresample=24000" in chain
    assert "asetrate=48000*1.05" not in chain
