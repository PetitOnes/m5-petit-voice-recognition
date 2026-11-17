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
