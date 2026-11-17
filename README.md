# M5 Petit Voice Recognition

## [English Page](./README_en.md)

GPU付きPC上で動作する音声認識(ASR)・話者識別・音声特徴量抽出サーバー。M5 Petitからの発話を、常時ONの軽量PCから呼び出すためのHTTP APIとして提供する。

MCP経由ではなく素のHTTP APIにしているのは、音声認識は応答速度が重要なため、MCPのオーバーヘッドを避けたいから。[m5-petit-speech](https://github.com/PetitOnes/m5-petit-speech)（TTS側）と対になるASRサーバー。

## 主な機能

- 音声の文字起こし（Whisper / [faster-whisper](https://github.com/SYSTRAN/faster-whisper)）
- 話者識別（[resemblyzer](https://github.com/resemble-ai/Resemblyzer)による声紋照合。任意のIDで登録・照合できる）
- 音声特徴量抽出（F0, MFCC, jitter, shimmer, HNRなど。librosa / parselmouth）
- 環境音の分類（現状はダミー実装。将来的にPANNs/YAMNetに差し替え予定）
- FastAPIによるHTTP API提供

## 構成

```
src/m5_petit_voice_recognition/
├── main.py                FastAPIエントリポイント
├── config.py              設定(.env読み込み)
├── schemas.py             APIレスポンス定義
├── api/
│   └── routes.py          エンドポイント定義
├── services/
│   ├── whisper_service.py        音声認識
│   ├── speaker_service.py        話者識別(resemblyzer)
│   ├── sound_event_service.py    環境音分類(現状ダミー)
│   ├── voice_feature_service.py  音声特徴量抽出
│   └── audio_loader.py           音声読み込み
└── utils/
    └── voice_summary.py   Claude向け要約生成
```

## 必要環境

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- GPUで動かす場合: CUDA（faster-whisperのCTranslate2バックエンド用。`libcublas.so.12`が必要）
  - CPUのみでも動作する（`WHISPER_DEVICE=cpu`）

## セットアップ

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv未インストールの場合

git clone https://github.com/PetitOnes/m5-petit-voice-recognition.git
cd m5-petit-voice-recognition
uv sync
cp .env.example .env
```

## 起動

```bash
uv run uvicorn m5_petit_voice_recognition.main:app --host 0.0.0.0 --port 8767
```

ポートは任意（デプロイ例では8767を使用）。

## API一覧

### `GET /health`

ヘルスチェック。

### `GET /help`

API一覧と使用例をテキストで返す。

### `POST /transcribe`

音声 → テキスト（話者識別付き）。

```bash
curl -X POST http://127.0.0.1:8767/transcribe -F "file=@sample.wav"
```

### `POST /extract_voice_features`

音声 → 音声特徴量（F0, MFCC, jitter, shimmer, HNRなど）。

### `POST /analyze_audio`

音声 → まとめて解析（メインAPI。文字起こし＋環境音＋音声特徴量＋話者識別）。

### `POST /analyze_audio_summary`

Claude向けの簡潔な要約（環境・ピッチ・話速などをラベル化して返す）。

### `POST /register_speaker`

話者の声を登録する（フォーム: `speaker_id`, `file`）。同一IDで再登録すると埋め込みが移動平均で更新される。

```bash
curl -X POST http://127.0.0.1:8767/register_speaker \
  -F "speaker_id=alice" -F "file=@alice_voice.wav"
```

### `GET /speakers`

登録済み話者IDの一覧を返す。

### `DELETE /speakers/{speaker_id}`

登録済み話者を削除する。

## 出力例（`/analyze_audio`）

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

## 環境変数

詳細は `.env.example` を参照。主なもの:

| 変数名 | 説明 | デフォルト |
| --- | --- | --- |
| `WHISPER_MODEL` | faster-whisperのモデルサイズ(tiny/base/small/medium/large-v3など) | `small` |
| `WHISPER_MODELS` | 選べるモデル(カンマ区切り)。リクエストのフォーム項目 `model` で選ぶ。`kotoba-v2` = kotoba-tech/kotoba-whisper-v2.0-faster(日本語向け)。`GET /models` で一覧 | `small,medium,large-v3-turbo,large-v3,kotoba-v2` |
| `WHISPER_VAD` | 声の無い区間を先に切る(Silero VAD) | `true` |
| `WHISPER_NO_SPEECH_THRESHOLD` / `WHISPER_LOGPROB_THRESHOLD` | 声が無さそう(この値より上)かつ自信が無い(この値より下)区間を捨てる | `0.6` / `-1.0` |
| `WHISPER_BLOCKLIST` | 無音・雑音で Whisper が作りがちな決まり文句(カンマ区切り)。それだけの区間を捨てる(ふつうの文の中にあれば残す)。捨てた文は `dropped` に | `ご視聴ありがとうございました,…,字幕` |
| `WHISPER_DEVICE` | `cpu` / `cuda` | `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8`(CPU) / `float16`(GPU)など | `int8` |
| `WHISPER_LANGUAGE` | 認識言語 | `ja` |
| `HOST` / `PORT` / `LOG_LEVEL` | 現状 `main.py` はuvicornに直接ホスト/ポートを渡す運用のため未使用(将来のCLI起動用に予約) | `0.0.0.0` / `8000` / `info` |
| `SOUND_MODEL` / `SOUND_TOP_K` / `SOUND_MIN_SCORE` | 環境音分類の設定(現状はダミー出力にのみ影響、実モデル未接続) | `panns` / `5` / `0.2` |
| `VOICE_FRAME_LENGTH` / `VOICE_HOP_LENGTH` | 音声特徴量抽出のフレーム/ホップ長 | `2048` / `256` |
| `VOICE_FMIN` / `VOICE_FMAX` | F0推定の探索範囲(Hz) | `65.0` / `400.0` |
| `VOICE_MFCC_N` | MFCC次元数 | `13` |
| `VOICE_PAUSE_DB_THRESHOLD` / `VOICE_MIN_PAUSE_SEC` | 無音区間(ポーズ)検出のしきい値 | `25.0` / `0.20` |
| `ENABLE_OPENSMILE` | 現状未使用(将来のeGeMAPS特徴量抽出用に予約) | `false` |
| `SPEAKER_REGISTRY_PATH` | 話者埋め込みの保存先JSONパス | `speaker_registry.json` |
| `SPEAKER_THRESHOLD` | 話者識別の類似度しきい値(これ未満は`unknown`) | `0.75` |

## CUDA設定（GPUで動かす場合）

必要条件:
- `libcublas.so.12`
- libcudnn（必要に応じて）

認識されない場合:

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcublas/12:$LD_LIBRARY_PATH
```

永続化:

```bash
echo '/usr/lib/x86_64-linux-gnu/libcublas/12' | sudo tee /etc/ld.so.conf.d/libcublas-12.conf
sudo ldconfig
```

## テスト

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

`tests/test_health.py` はCPUのみで動作する。`tests/test_analyze_audio.py` は現状プレースホルダ(未実装)で、GPU依存のテストは含まれていない。

## systemd化（例）

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

## 別PCからアクセス

Tailscale経由を想定:

```bash
curl http://100.xxx.xxx.xxx:8767/health
```

## テスト用音声の作成

```bash
arecord -d 5 -f S16_LE -r 16000 -c 1 sample.wav
```

## 開発メモ

- Whisperは最初CPUで動作確認するのが安全。GPUは後から有効化する
- 音声は1回だけロードして各処理(ASR/環境音/特徴量/話者識別)に分岐する
- 環境音分類・eGeMAPS特徴量は現状プレースホルダで、実モデルは未接続

## 今後の拡張

- PANNs / YAMNetによる環境音分類の実装
- openSMILE(eGeMAPS)特徴量の接続
- ROS2ノード化
- リアルタイム処理

## 備考

このサーバーは「音声理解エンジン」として設計されている。Claudeやロボットと組み合わせて使用する前提で、[m5-petit-mcp](https://github.com/PetitOnes/m5-petit-mcp)の`VOICE_API_HOST`から呼び出される想定。
