# Cloud Run GPU で LLM（Ollama/Qwen）と文字起こし（faster-whisper）を
# 1コンテナで提供する。どちらもOpenAI互換のエンドポイントとして出す。
#
#   docker build --build-arg MODEL=qwen3:30b-a3b --build-arg WHISPER_MODEL=large-v3 .
#
# モデルは**両方ともイメージに焼き込む**。Cloud Runはイメージを
# ストリーミング展開するため、起動時にネットワーク越しで取得するより速い。

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG MODEL=qwen3:30b-a3b
ARG WHISPER_MODEL=large-v3

ENV MODEL=${MODEL} \
    WHISPER_MODEL=${WHISPER_MODEL} \
    WHISPER_LANGUAGE=ja \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE=int8_float16 \
    OLLAMA_MODELS=/models/ollama \
    HF_HOME=/models/hf \
    OLLAMA_INTERNAL_URL=http://127.0.0.1:11434 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        "faster-whisper>=1.0" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "python-multipart>=0.0.9"

# Ollama本体
RUN curl -fsSL https://ollama.com/install.sh | sh

# --- モデルをイメージへ焼き込む ---
# Whisper（CTranslate2形式に変換済みのものが取得される）
RUN mkdir -p /models/hf && python3 -c "\
import os; from faster_whisper import WhisperModel; \
WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8'); \
print('whisper cached')"

# LLM（ollama serve を一時起動して pull する）
RUN mkdir -p /models/ollama && \
    (OLLAMA_HOST=127.0.0.1:11434 ollama serve &) && \
    timeout 300 sh -c 'until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do sleep 2; done' && \
    ollama pull ${MODEL} && \
    pkill -f "ollama serve" || true

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8080
# --timeout-keep-alive を伸ばす（長い音声のアップロードと長文生成のため）
CMD ["python3", "-m", "uvicorn", "server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "600"]
