"""決まり文句・声の無い区間を捨てる決まり(モデルは読み込まない)"""

from m5_petit_voice_recognition.services import whisper_service as ws


def test_drops_outro_phrases():
    assert ws.keep_segment("ご視聴ありがとうございました", 0.1, -0.2) == "blocklist"
    assert ws.keep_segment(" ご視聴ありがとうございました。", 0.1, -0.2) == "blocklist"
    assert ws.keep_segment("字幕", 0.1, -0.2) == "blocklist"


def test_keeps_real_speech():
    assert ws.keep_segment("ありがとうございました", 0.1, -0.2) is None   # ふつうのお礼は残す
    assert ws.keep_segment("きょうは字幕つきの映画を見たよ", 0.1, -0.2) is None
    assert ws.keep_segment("おはよう", 0.7, -0.3) is None   # 自信があれば残す


def test_drops_no_speech_low_confidence():
    assert ws.keep_segment("あ", 0.9, -1.5) == "no_speech"
    assert ws.keep_segment("   ", 0.0, 0.0) == "empty"


def test_model_choice():
    svc = ws.WhisperService()
    assert svc.resolve(None) == ws.settings.whisper_model
    assert svc.resolve("kotoba-v2") == "kotoba-v2"
    try:
        svc.resolve("evil/model")
        raise AssertionError
    except ValueError:
        pass


def test_max_loaded_keeps_recent(monkeypatch):
    import sys
    import types

    made = []
    fake = types.SimpleNamespace(WhisperModel=lambda mid, **kw: made.append(mid) or mid)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setattr(ws.settings, "whisper_max_loaded", 2)
    svc = ws.WhisperService()
    for n in ("small", "kotoba-v2", "small", "large-v3-turbo"):
        svc._get_model(n)
    assert svc.loaded() == ["small", "large-v3-turbo"]   # kotoba-v2 がいちばん長く使われていなかった
    assert len(made) == 3
