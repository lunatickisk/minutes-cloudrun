#!/usr/bin/env bash
# Cloud Run GPU に LLM（Qwen）＋文字起こし（faster-whisper）を1サービスでデプロイする。
#
#   bash deploy/cloudrun/deploy.sh
#   MODEL=qwen3:14b WHISPER_MODEL=medium bash deploy/cloudrun/deploy.sh
#
# ゼロスケールするので、リクエストが無い間はGPU課金が発生しません。
set -euo pipefail

MODEL="${MODEL:-qwen3:30b-a3b}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_DEVICE="${WHISPER_DEVICE:-cuda}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-minutes-inference}"
REPO="${REPO:-llm}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "プロジェクトIDが未設定です: gcloud config set project PROJECT_ID" >&2
  exit 1
fi

TAG="${MODEL//:/-}_${WHISPER_MODEL}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${TAG//\//-}"

cat <<EOF
=========================================================
 Cloud Run GPU デプロイ（LLM + 文字起こし 同居）
   project : ${PROJECT}
   region  : ${REGION}
   service : ${SERVICE}
   LLM     : ${MODEL}
   Whisper : ${WHISPER_MODEL} (${WHISPER_DEVICE})
=========================================================
 GPU: NVIDIA L4 (24GB)
   ${MODEL}  約19〜22GB
   Whisper(int8)  約1.5〜2GB
   合計が24GBに近いため、OOM時は自動でWhisperをCPUへ退避します。
 課金: リクエスト処理中のみ（ゼロスケール）
=========================================================
EOF

echo "[1/4] APIを有効化"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com --project "${PROJECT}"

gcloud artifacts repositories describe "${REPO}" \
  --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location "${REGION}" \
  --description="inference images" --project "${PROJECT}"

echo "[2/4] イメージをビルド（モデルを2つ焼き込むため時間がかかります）"
gcloud builds submit "$(dirname "$0")" \
  --tag "${IMAGE}" \
  --project "${PROJECT}" \
  --timeout=7200s \
  --machine-type=e2-highcpu-32 \
  --disk-size=200

echo "[3/4] Cloud Run にデプロイ"
gcloud beta run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 8 \
  --memory 32Gi \
  --concurrency 1 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 3600 \
  --no-cpu-throttling \
  --no-allow-unauthenticated \
  --port 8080 \
  --set-env-vars "MODEL=${MODEL},WHISPER_MODEL=${WHISPER_MODEL},WHISPER_DEVICE=${WHISPER_DEVICE},OLLAMA_KEEP_ALIVE=5m"

URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" --project "${PROJECT}" --format='value(status.url)')"

echo "[4/4] 呼び出し権限を付与"
ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [[ -n "${ACCOUNT}" && "${ACCOUNT}" != "(unset)" ]]; then
  gcloud run services add-iam-policy-binding "${SERVICE}" \
    --region "${REGION}" --project "${PROJECT}" \
    --member "user:${ACCOUNT}" --role roles/run.invoker >/dev/null
  echo "  ${ACCOUNT} に roles/run.invoker を付与しました"
fi

cat <<EOF

✓ デプロイ完了

  service_url: ${URL}

config.yaml に次を設定してください（LLMと文字起こしで同じURLです）。

transcription:
  provider: whisper_remote
  whisper_remote:
    service_url: "${URL}"
    auth: iam

llm:
  provider: cloudrun
  model: "${MODEL}"
  cloudrun:
    service_url: "${URL}"
    auth: iam

疎通確認（GPUに載ったかをここで確認できます）:
  curl -H "Authorization: Bearer \$(gcloud auth print-identity-token)" ${URL}/healthz
  python run.py check-llm

実行:
  python run.py ingest data/sample/amagasaki_2011_05_18.json --auto-approve

注意:
  初回リクエストはモデル読み込みのため2〜5分かかることがあります。
  healthz の whisper.device が "cpu" になっていたらVRAM不足でフォールバックしています。
  その場合は WHISPER_MODEL=medium か MODEL=qwen3:14b で再デプロイしてください。
EOF
