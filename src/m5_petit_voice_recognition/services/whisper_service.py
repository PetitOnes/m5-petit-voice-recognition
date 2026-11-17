from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path

import soundfile as sf
from starlette.datastructures import UploadFile

from m5_petit_voice_recognition.config import settings

# 短い名前 → faster-whisper に渡す id。ここに無い名前は、そのまま faster-whisper / Hugging Face の id として使う
MODELS = {
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
    "kotoba-v2": "kotoba-tech/kotoba-whisper-v2.0-faster",   # 日本語向けに蒸留した Whisper
}


def _norm(text: str) -> str:
    return re.sub(r"[\s、。!?！？・…]+", "", text)


def allowed_models() -> list[str]:
    return [m.strip() for m in settings.whisper_models.split(",") if m.strip()]


def blocklist() -> list[str]:
    return [_norm(b) for b in settings.whisper_blocklist.split(",") if _norm(b)]


def keep_segment(text: str, no_speech_prob: float | None, avg_logprob: float | None) -> str | None:
    """残さない区間なら、その理由。残すなら None。
    - 声が無さそうで(no_speech_prob が高い)、しかも自信が無い(avg_logprob が低い)区間
    - 決まり文句だけの区間(Whisper が無音・雑音で作りがちな字幕の文)"""
    if (no_speech_prob is not None and avg_logprob is not None
            and no_speech_prob > settings.whisper_no_speech_threshold
            and avg_logprob < settings.whisper_logprob_threshold):
        return "no_speech"
    n = _norm(text)
    if not n:
        return "empty"
    if any(n == b or (b in n and len(n) <= len(b) + 4) for b in blocklist()):
        return "blocklist"
    return None


class WhisperService:
    def __init__(self) -> None:
        self._models: dict = {}
        self._lock = threading.Lock()

    def resolve(self, name: str | None) -> str:
        """使うモデルの名前(空なら WHISPER_MODEL)。選べないものは ValueError"""
        name = (name or settings.whisper_model).strip()
        if name != settings.whisper_model and name not in allowed_models():
            raise ValueError(f"model は {', '.join([settings.whisper_model] + allowed_models())} のどれか")
        return name

    def _get_model(self, name: str):
        from faster_whisper import WhisperModel

        with self._lock:
            if name not in self._models:
                print("load whisper:", name, settings.whisper_device, settings.whisper_compute_type)
                self._models[name] = WhisperModel(
                    MODELS.get(name, name),
                    device=settings.whisper_device,
                    compute_type=settings.whisper_compute_type,
                )
            return self._models[name]

    def loaded(self) -> list[str]:
        return list(self._models)

    def transcribe_path(self, audio_path: Path, model: str | None = None) -> dict:
        info = sf.info(str(audio_path))
        name = self.resolve(model)
        segments, meta = self._get_model(name).transcribe(
            str(audio_path),
            language=settings.whisper_language or None,
            vad_filter=settings.whisper_vad,
            condition_on_previous_text=False,   # 前の文につられて決まり文句を続けない
        )
        kept: list[str] = []
        dropped: list[dict] = []
        for seg in segments:
            why = keep_segment(seg.text, getattr(seg, "no_speech_prob", None), getattr(seg, "avg_logprob", None))
            if why:
                dropped.append({"text": seg.text.strip(), "why": why})
            else:
                kept.append(seg.text)
        return {
            "text": "".join(kept).strip(),
            "language": getattr(meta, "language", None),
            "duration_sec": float(info.duration),
            "model": name,
            "dropped": dropped,
        }

    async def transcribe_upload(self, upload_file: UploadFile, model: str | None = None) -> dict:
        suffix = Path(upload_file.filename or "audio.wav").suffix or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload_file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            return self.transcribe_path(tmp_path, model)
        finally:
            tmp_path.unlink(missing_ok=True)
