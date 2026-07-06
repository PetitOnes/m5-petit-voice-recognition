# M5 Petit Voice Recognition

## [日本語ページ](./README.md)

A speech-processing server (ASR, speaker identification, and voice feature extraction) that runs on a GPU-equipped PC, exposed as an HTTP API so an always-on lightweight PC can call it to process what an M5 Petit hears.

It's a plain HTTP API rather than MCP because ASR needs to respond quickly and we want to avoid MCP overhead. This is the ASR counterpart to [m5-petit-speech](https://github.com/PetitOnes/m5-petit-speech) (the TTS side).

## Main features

- Speech-to-text transcription (Whisper / [faster-whisper](https://github.com/SYSTRAN/faster-whisper))
- Speaker identification ([resemblyzer](https://github.com/resemble-ai/Resemblyzer) voiceprint matching; register and match against arbitrary speaker IDs)
- Voice feature extraction (F0, MFCC, jitter, shimmer, HNR, etc. via librosa / parselmouth)
- Sound event classification (currently a stub; PANNs/YAMNet planned)
- HTTP API via FastAPI

## Directory layout

```
src/m5_petit_voice_recognition/
├── main.py                FastAPI entry point
├── config.py              Settings (.env loading)
├── schemas.py             API response definitions
├── api/
│   └── routes.py          Endpoint definitions
├── services/
│   ├── whisper_service.py        Speech recognition
│   ├── speaker_service.py        Speaker identification (resemblyzer)
│   ├── sound_event_service.py    Sound event classification (currently a stub)
│   ├── voice_feature_service.py  Voice feature extraction
│   └── audio_loader.py           Audio loading
└── utils/
    └── voice_summary.py   Builds a summary for Claude
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- CUDA if running on GPU (needed by faster-whisper's CTranslate2 backend; requires `libcublas.so.12`)
  - Also works CPU-only (`WHISPER_DEVICE=cpu`)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't installed yet

git clone https://github.com/PetitOnes/m5-petit-voice-recognition.git
cd m5-petit-voice-recognition
uv sync
cp .env.example .env
```

## Running

```bash
uv run uvicorn m5_petit_voice_recognition.main:app --host 0.0.0.0 --port 8767
```

The port is arbitrary (the reference deployment uses 8767).

## API list

### `GET /health`

Health check.

### `GET /help`

Returns the API list and usage examples as text.

### `POST /transcribe`

Audio → text (includes speaker identification).

```bash
curl -X POST http://127.0.0.1:8767/transcribe -F "file=@sample.wav"
```

### `POST /extract_voice_features`

Audio → voice features (F0, MFCC, jitter, shimmer, HNR, etc.).

### `POST /analyze_audio`

Audio → full analysis (main API: transcription + sound events + voice features + speaker identification).

### `POST /analyze_audio_summary`

A compact summary for Claude (environment, pitch, speech rate, etc. as labels).

### `POST /register_speaker`

Register a speaker's voice (form fields: `speaker_id`, `file`). Re-registering the same ID updates the embedding via a running average.

```bash
curl -X POST http://127.0.0.1:8767/register_speaker \
  -F "speaker_id=alice" -F "file=@alice_voice.wav"
```

### `GET /speakers`

List registered speaker IDs.

### `DELETE /speakers/{speaker_id}`

Remove a registered speaker.

## Example output (`/analyze_audio`)

```json
{
  "transcript": "こんにちは",
  "language": "ja",
  "duration_sec": 2.3,
  "sound_events": [{"label": "Speech", "score": 0.9}],
  "voice_features": {"f0_mean_hz": 210.5, "mfcc_mean": ["..."]},
  "summary_for_claude": "環境音: Speech(0.90)。音声特徴: 平均F0は210.5 Hz...",
  "speaker": "alice",
  "speaker_confidence": 0.82
}
```

## Environment variables

See `.env.example` for the full list. Notable ones:

| Variable | Description | Default |
| --- | --- | --- |
| `WHISPER_MODEL` | faster-whisper model size (tiny/base/small/medium/large-v3, etc.) | `small` |
| `WHISPER_DEVICE` | `cpu` / `cuda` | `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8` (CPU) / `float16` (GPU), etc. | `int8` |
| `WHISPER_LANGUAGE` | Transcription language | `ja` |
| `HOST` / `PORT` / `LOG_LEVEL` | Currently unused — `main.py` is run via uvicorn with explicit host/port flags. Reserved for a future CLI entrypoint | `0.0.0.0` / `8000` / `info` |
| `SOUND_MODEL` / `SOUND_TOP_K` / `SOUND_MIN_SCORE` | Sound event classification settings (currently only affects the stub output; no real model is wired up yet) | `panns` / `5` / `0.2` |
| `VOICE_FRAME_LENGTH` / `VOICE_HOP_LENGTH` | Frame/hop length for voice feature extraction | `2048` / `256` |
| `VOICE_FMIN` / `VOICE_FMAX` | F0 search range (Hz) | `65.0` / `400.0` |
| `VOICE_MFCC_N` | Number of MFCC coefficients | `13` |
| `VOICE_PAUSE_DB_THRESHOLD` / `VOICE_MIN_PAUSE_SEC` | Thresholds for detecting silent pauses | `25.0` / `0.20` |
| `ENABLE_OPENSMILE` | Currently unused; reserved for future eGeMAPS feature extraction | `false` |
| `SPEAKER_REGISTRY_PATH` | Path to the JSON file storing speaker embeddings | `speaker_registry.json` |
| `SPEAKER_THRESHOLD` | Similarity threshold for speaker identification (below this, returns `unknown`) | `0.75` |

## CUDA configuration (for GPU use)

Requirements:
- `libcublas.so.12`
- libcudnn (if needed)

If it isn't detected:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcublas/12:$LD_LIBRARY_PATH
```

To persist it:

```bash
echo '/usr/lib/x86_64-linux-gnu/libcublas/12' | sudo tee /etc/ld.so.conf.d/libcublas-12.conf
sudo ldconfig
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

`tests/test_health.py` runs on CPU only. `tests/test_analyze_audio.py` is currently a placeholder (not yet implemented); there are no GPU-dependent tests at this time.

## systemd (example)

```ini
[Unit]
Description=M5 Petit Voice Recognition Server
After=network.target

[Service]
User=<YOUR_USER>
WorkingDirectory=/path/to/m5-petit-voice-recognition
Environment="PYTHONUNBUFFERED=1"
ExecStart=/path/to/.local/bin/uv run uvicorn m5_petit_voice_recognition.main:app --host 0.0.0.0 --port 8767
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Accessing from another machine

Assumes access over Tailscale:

```bash
curl http://100.xxx.xxx.xxx:8767/health
```

## Creating a test audio file

```bash
arecord -d 5 -f S16_LE -r 16000 -c 1 sample.wav
```

## Development notes

- It's safest to verify Whisper on CPU first, then enable GPU afterward
- Audio is loaded once and then branched out to each processing step (ASR / sound events / features / speaker ID)
- Sound event classification and eGeMAPS features are currently placeholders; no real model is wired up yet

## Planned extensions

- Sound event classification via PANNs / YAMNet
- Wire up openSMILE (eGeMAPS) features
- ROS2 node
- Real-time processing

## Notes

This server is designed as a "voice understanding engine", intended to be used together with Claude or a robot — called from [m5-petit-mcp](https://github.com/PetitOnes/m5-petit-mcp)'s `VOICE_API_HOST`.
