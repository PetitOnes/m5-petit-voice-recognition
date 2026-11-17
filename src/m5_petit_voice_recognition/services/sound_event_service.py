"""音の種類(AudioSet の 527 種類)を見分ける。PANNs の Cnn14_DecisionLevelMax(何秒目に何が鳴ったかも出せる)。

- 重み(約 300 MB)は初回に ~/panns_data/ へ取ってくる(SOUND_CHECKPOINT で場所を変えられる)
- SOUND_MODEL=none で使わない(何も返さない)。読み込めないときも何も返さない(聞き取りは止めない)
- 声の文字起こし(Whisper)とは別。同じ音を両方に通す(routes.py)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from m5_petit_voice_recognition.config import settings

log = logging.getLogger(__name__)

PANNS_SR = 32000
FPS = 100                 # Cnn14_DecisionLevelMax は 10 ms ごと
MIN_SEC = 1.0             # これより短い音は無音で埋める
URL = "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"


def _checkpoint() -> Path:
    return Path(settings.sound_checkpoint or Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth")


class SoundEventService:
    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._device = "cpu"
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        if settings.sound_model.lower() != "panns" or self._failed:
            return None
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from panns_inference import labels
                from panns_inference.models import Cnn14_DecisionLevelMax

                ck = _checkpoint()
                if not ck.exists() or ck.stat().st_size < 3e8:
                    ck.parent.mkdir(parents=True, exist_ok=True)
                    log.warning("PANNs の重みを取ってくる: %s", ck)
                    os.system(f'wget -q -O "{ck}" "{URL}"')   # panns_inference と同じ取り方
                self._device = "cuda" if settings.sound_device == "cuda" and torch.cuda.is_available() else "cpu"
                model = Cnn14_DecisionLevelMax(sample_rate=PANNS_SR, window_size=1024, hop_size=320, mel_bins=64,
                                               fmin=50, fmax=14000, classes_num=len(labels))
                state = torch.load(str(ck), map_location=self._device, weights_only=False)
                model.load_state_dict(state["model"])
                model.to(self._device).eval()
                self._model, self._labels = model, list(labels)
            except Exception as e:   # noqa: BLE001  音の種類が分からなくても、聞き取りは続ける
                log.warning("PANNs を読み込めない: %s", e)
                self._failed = True
                return None
            return self._model

    def framewise(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """(フレーム数, 527) の確からしさ。10 ms ごと。使えなければ None"""
        model = self._load()
        if model is None or waveform is None or len(waveform) == 0:
            return None
        import librosa
        import torch

        x = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sample_rate, target_sr=PANNS_SR)
        n = len(x)
        if n < PANNS_SR * MIN_SEC:
            x = np.pad(x, (0, int(PANNS_SR * MIN_SEC) - n))
        with torch.no_grad():
            out = model(torch.from_numpy(x[None, :]).to(self._device), None)["framewise_output"]
        fw = out[0].cpu().numpy()
        return fw[: max(1, round(n / PANNS_SR * FPS))]

    def classify_waveform(self, waveform: np.ndarray, sample_rate: int) -> list[dict]:
        """音全体で、確からしい種類(上から SOUND_TOP_K、SOUND_MIN_SCORE 以上)"""
        fw = self.framewise(waveform, sample_rate)
        if fw is None:
            return []
        clip = fw.max(axis=0)
        idx = np.argsort(clip)[::-1][: settings.sound_top_k]
        return [{"label": self._labels[i], "score": round(float(clip[i]), 3)} for i in idx if clip[i] >= settings.sound_min_score]

    def timeline(self, waveform: np.ndarray, sample_rate: int, step_sec: float = 1.0, top: int = 3) -> list[dict]:
        """step_sec ごとに、確からしい種類(SOUND_MIN_SCORE 以上、上から top)。[{"t": 0, "labels": [{"label", "score"}]}]"""
        fw = self.framewise(waveform, sample_rate)
        if fw is None:
            return []
        hop = max(1, int(FPS * step_sec))
        out = []
        for k, s in enumerate(range(0, len(fw), hop)):
            seg = fw[s:s + hop].max(axis=0)
            idx = np.argsort(seg)[::-1][:top]
            out.append({"t": round(k * step_sec, 2),
                        "labels": [{"label": self._labels[i], "score": round(float(seg[i]), 3)} for i in idx if seg[i] >= settings.sound_min_score]})
        return out

    @property
    def ready(self) -> bool:
        return self._model is not None
