"""音の種類と音のかたち(モデルを読まずに確かめられるところ)"""

import numpy as np

from m5_petit_voice_recognition.api import routes
from m5_petit_voice_recognition.services import sound_event_service as ses


def test_shape_numbers_only():
    sr = 16000
    t = np.arange(sr * 3) / sr
    x = np.concatenate([np.zeros(sr), 0.3 * np.sin(2 * np.pi * 1000 * t[: sr * 2])])
    s = routes._shape(x, sr, 1.0)
    assert [r["t"] for r in s] == [0.0, 1.0, 2.0]
    assert s[0]["db"] < -100 and s[1]["db"] > -15
    assert 800 < s[1]["centroid"] < 1200
    assert set(s[0]) == {"t", "db", "centroid", "onset"}   # 言葉は付けない


def test_no_model_returns_nothing(monkeypatch):
    monkeypatch.setattr(ses.settings, "sound_model", "none")
    svc = ses.SoundEventService()
    x = np.zeros(16000)
    assert svc.classify_waveform(x, 16000) == [] and svc.timeline(x, 16000) == []
