# Cloud Run GPU で LLM と文字起こしを動かす

1つのCloud Runサービスに **Qwen（Ollama）と faster-whisper を同居**させ、
どちらもOpenAI互換のエンドポイントとして提供します。

```
POST /v1/chat/completions      → Qwen（Ollama へプロキシ）
POST /v1/audio/transcriptions  → faster-whisper
GET  /healthz                  → 両方の状態とGPU割り当て
```

クライアント側は **LLMも文字起こしも同じURL** を指すだけです。

## なぜ Cloud Run か

| | Cloud Run GPU | Vertex 自前デプロイ |
|---|---|---|
| 課金 | **リクエスト処理中のみ**（ゼロスケール） | GPU起動時間ずっと |
| 消し忘れリスク | **低い**（勝手に止まる） | 高い |
| GPU | NVIDIA L4 24GB | 選択可 |
| 載るモデル | 〜30B級（**本番のRTX5080/5090と同等**） | 80B以上も可 |

## 手順

```bash
bash deploy/cloudrun/deploy.sh

# 小さめの構成にする場合
MODEL=qwen3:14b WHISPER_MODEL=medium bash deploy/cloudrun/deploy.sh
```

完了時に表示される URL を `config.yaml` に設定します（**両方とも同じURL**）。

```yaml
transcription:
  provider: whisper_remote
  whisper_remote:
    service_url: "https://minutes-inference-xxxx-uc.a.run.app"
    auth: iam

llm:
  provider: cloudrun
  model: "qwen3:30b-a3b"
  cloudrun:
    service_url: "https://minutes-inference-xxxx-uc.a.run.app"
    auth: iam
```

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ${URL}/healthz
python run.py check-llm
python run.py ingest data/sample/amagasaki_2011_05_18.json --auto-approve
```

## VRAM配分（ここが一番の勘所）

L4は24GB。両方をGPUに載せると余裕は小さくなります。

```
Qwen3-30B-A3B (Q4)        約19〜22GB
Whisper large-v3 (int8)   約1.5〜2GB
--------------------------------
合計                      約21〜24GB
```

そのため次の3つの安全策を入れてあります。

**1. Whisperは `int8_float16`。** float16の約3GBに対し1.5〜2GBに収まります。
日本語の認識精度への影響は小さく、VRAMの節約効果の方が大きいと判断しました。

**2. OOM時はCPUへ自動フォールバック。** cudaでのロードに失敗しても
サービスは落ちず、遅くなるだけで動き続けます。

**3. `OLLAMA_KEEP_ALIVE=5m`。** アイドル時にLLMをVRAMから降ろします。
パイプラインは「文字起こし1回 → LLM多数回」の順に走るため、
文字起こし中はLLMがVRAMを掴んでいる必要がありません。

### 実際にGPUに載ったかの確認

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" ${URL}/healthz
```

```json
{"ok": true,
 "llm": {"ok": true, "default_model": "qwen3:30b-a3b"},
 "whisper": {"device": "cuda", "compute_type": "int8_float16", "loaded": true}}
```

`whisper.device` が `"cpu"` になっていたらVRAM不足でフォールバックしています。
GPUで動かしたい場合は次のいずれかで再デプロイしてください。

```bash
WHISPER_MODEL=medium bash deploy/cloudrun/deploy.sh   # Whisperを小さく
MODEL=qwen3:14b      bash deploy/cloudrun/deploy.sh   # LLMを小さく（約9GB）
```

## 設計上の判断

**モデルは両方ともイメージに焼き込んでいます。** Cloud Run はイメージを
ストリーミング展開するため、起動時にネットワーク越しで取得するより速く済みます。
ビルドに時間がかかる代わりに、コールドスタートが実用的な範囲に収まります。
モデルを変える場合はイメージの作り直しになります。

**`--concurrency 1`。** 推論はGPUを占有するため、同時リクエストを受けても
速くならずVRAMを圧迫するだけです。

**`--max-instances 1`。** 暴走的なスケールアウトでGPU課金が膨らむのを防ぎます。
チームで同時利用するなら増やしてください。

**`--no-allow-unauthenticated`。** 議事録には機密情報が入りうるため公開していません。
クライアントは**IDトークン**（アクセストークンではない）を付与します。

## 注意点

**初回リクエストは2〜5分**かかります（コールドスタートで2つのモデルを読み込むため）。
常時応答が必要なら `--min-instances 1` にできますが、
**その瞬間からGPU課金が常時発生**しゼロスケールの利点が消えます。

**L4のクォータ**が必要な場合があります
（Cloud Run の「Total Nvidia L4 GPU allocation, per project per region」）。

**80Bは載りません。** MoEで実効3Bでも全エキスパートの重みをVRAMに置くためです。

## 片付け

ゼロスケールするので通常は放置で構いませんが、完全に消す場合:

```bash
gcloud run services delete minutes-inference --region us-central1
```
