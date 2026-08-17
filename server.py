#!/usr/bin/env python3
"""Cloud Run 上で LLM と文字起こしを1サービスとして提供するゲートウェイ。

  POST /v1/chat/completions      → Ollama（Qwen等）へプロキシ
  POST /v1/audio/transcriptions  → faster-whisper で処理
  GET  /v1/models                → Ollama のモデル一覧
  GET  /healthz                  → 両方の状態

どちらもOpenAI互換の形にしてあるので、将来 OpenAI / Groq / vLLM などに
差し替えてもクライアント側を変えずに済む。

--------------------------------------------------------------------------
GPUの割り当てについて（重要）
--------------------------------------------------------------------------
L4は24GB。既定では **両方ともGPUに載せる**。

    Qwen3-30B-A3B (Q4)        約19〜22GB
    Whisper large-v3 (int8)   約1.5〜2GB   ← float16の約3GBより節約できる
    ------------------------------------
    合計                      約21〜24GB  ← 収まるが余裕は小さい

余裕が小さいため、次の2つの安全策を入れてある。

  1. Whisperは既定で int8_float16。精度への影響は小さくVRAMを半分に抑えられる
  2. cudaでのロードに失敗（OOM）したら**自動でCPUへフォールバック**する。
     サービスが落ちるのではなく、遅くなるだけで動き続ける

OOMが頻発する場合は次のいずれかで回避できる。
  - WHISPER_DEVICE=cpu           文字起こしだけCPUに逃がす
  - WHISPER_MODEL=medium         Whisperを小さくする
  - MODEL=qwen3:14b              LLMを小さくする（約9GB）
  - OLLAMA_KEEP_ALIVE=5m         アイドル時にLLMをVRAMから降ろす（既定）

パイプラインは「文字起こし1回 → LLM多数回」の順に走るため、
KEEP_ALIVEを有限にしておくと実際のピークVRAMはさらに下がる。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

OLLAMA_URL = os.environ.get("OLLAMA_INTERNAL_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("MODEL", "qwen3:30b-a3b")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda").lower()  # cuda | cpu | auto
#: GPU時の量子化。int8_float16 はfloat16の約半分のVRAMで精度低下は小さい
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8_float16")
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ja")

app = FastAPI(title="minutes-pipeline inference gateway")
_whisper = None
_whisper_info: dict[str, Any] = {"device": "", "compute_type": "", "loaded": False}
_ollama_proc: subprocess.Popen | None = None


# ---------------------------------------------------------------- Ollama

def _start_ollama() -> None:
    """Ollama をバックグラウンドで起動する。"""
    global _ollama_proc
    if _ollama_proc and _ollama_proc.poll() is None:
        return
    env = dict(os.environ)
    env["OLLAMA_HOST"] = OLLAMA_URL.replace("http://", "")
    # WhisperとGPUを共有するため、アイドル時はVRAMを解放する。
    # -1（常駐）にすると文字起こし側がVRAM不足になりやすい。
    env.setdefault("OLLAMA_KEEP_ALIVE", "5m")
    env.setdefault("OLLAMA_NUM_PARALLEL", "1")
    print(f"[startup] starting ollama on {env['OLLAMA_HOST']}", flush=True)
    _ollama_proc = subprocess.Popen(["ollama", "serve"], env=env)


def _wait_ollama(timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5):
                return True
        except Exception:
            time.sleep(2)
    return False


def _proxy(path: str, body: bytes | None, method: str = "POST") -> Response:
    url = f"{OLLAMA_URL}{path}"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        # LLM生成は長い。Cloud Runのリクエストタイムアウトに任せる
        with urllib.request.urlopen(req, timeout=3600) as r:
            return Response(
                content=r.read(),
                status_code=r.status,
                media_type=r.headers.get("Content-Type", "application/json"),
            )
    except urllib.error.HTTPError as e:
        return Response(content=e.read(), status_code=e.code, media_type="application/json")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"Ollamaに接続できません: {e}") from e


# ---------------------------------------------------------------- Whisper

def get_whisper():
    """モデルは初回に1回だけロードして使い回す。

    LLMと同じGPUを共有するため、cudaでのロードに失敗（VRAM不足）した場合は
    CPUへ自動フォールバックする。サービスを落とさず、遅くなるだけで済ませる。
    """
    global _whisper
    if _whisper is not None:
        return _whisper

    from faster_whisper import WhisperModel

    device = WHISPER_DEVICE
    if device == "auto":
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

    attempts = (
        [("cuda", WHISPER_COMPUTE), ("cpu", "int8")] if device == "cuda" else [("cpu", "int8")]
    )
    last_err: Exception | None = None
    for dev, compute in attempts:
        try:
            print(f"[startup] loading whisper {WHISPER_MODEL} on {dev}/{compute}", flush=True)
            started = time.time()
            _whisper = WhisperModel(WHISPER_MODEL, device=dev, compute_type=compute)
            _whisper_info.update(device=dev, compute_type=compute, loaded=True)
            print(f"[startup] whisper loaded in {time.time() - started:.1f}s", flush=True)
            return _whisper
        except Exception as e:
            last_err = e
            print(
                f"[warn] whisper を {dev} にロードできませんでした: {str(e)[:200]}",
                flush=True,
            )
            if dev == "cuda":
                print(
                    "[warn] VRAM不足の可能性があります。CPUで再試行します。"
                    "  恒久的に回避するには WHISPER_DEVICE=cpu か、"
                    "より小さい MODEL / WHISPER_MODEL を指定してください。",
                    flush=True,
                )
    raise RuntimeError(f"whisperをロードできません: {last_err}")


# ---------------------------------------------------------------- lifecycle

@app.on_event("startup")
def _warm() -> None:
    _start_ollama()
    if not _wait_ollama():
        print("[startup] ollama did not become ready in time", flush=True)
    try:
        get_whisper()
    except Exception as e:
        print(f"[startup] whisper load failed: {e}", flush=True)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ollama_ok = False
    models: list[str] = []
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            models = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
            ollama_ok = True
    except Exception:
        pass
    return {
        "ok": ollama_ok and _whisper_info["loaded"],
        "llm": {"ok": ollama_ok, "default_model": LLM_MODEL, "models": models},
        "whisper": {**_whisper_info, "model": WHISPER_MODEL},
    }


# ---------------------------------------------------------------- LLM

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return _proxy("/v1/chat/completions", await request.body())


@app.get("/v1/models")
def list_models() -> Response:
    return _proxy("/v1/models", None, method="GET")


# ---------------------------------------------------------------- 文字起こし

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=WHISPER_MODEL),
    language: str = Form(default=DEFAULT_LANGUAGE),
    prompt: str = Form(default=""),
    response_format: str = Form(default="verbose_json"),
    temperature: float = Form(default=0.0),
) -> Any:
    """OpenAI Audio API 互換。

    パイプラインは各発話の時刻を必要とするため既定を verbose_json にしている
    （OpenAIの既定 json はセグメントを返さない）。
    """
    try:
        whisper = get_whisper()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"モデルをロードできません: {e}") from e

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = Path(tempfile.mkdtemp()) / f"input{suffix}"
    tmp.write_bytes(await file.read())

    params: dict[str, Any] = {
        "language": language or None,
        "beam_size": int(os.environ.get("WHISPER_BEAM_SIZE", "5")),
        "vad_filter": os.environ.get("WHISPER_VAD", "1") == "1",
        "word_timestamps": True,
        "condition_on_previous_text": False,   # 長時間音声での幻覚の連鎖を防ぐ
        "hallucination_silence_threshold": 2.0,
        "temperature": temperature,
    }
    if prompt:
        params["initial_prompt"] = prompt

    started = time.time()
    try:
        segments, info = whisper.transcribe(str(tmp), **params)
        out = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(
                {
                    "id": len(out),
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": text,
                    "avg_logprob": float(getattr(s, "avg_logprob", 0.0)),
                    "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文字起こしに失敗しました: {e}") from e
    finally:
        try:
            tmp.unlink()
            tmp.parent.rmdir()
        except OSError:
            pass

    elapsed = time.time() - started
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    print(
        f"[transcribe] {file.filename} audio={duration:.0f}s proc={elapsed:.1f}s "
        f"({(duration / elapsed if elapsed else 0):.1f}x) segments={len(out)}",
        flush=True,
    )

    full_text = "".join(s["text"] for s in out)
    if response_format == "text":
        return JSONResponse(content=full_text, media_type="text/plain")
    if response_format == "json":
        return {"text": full_text}
    return {
        "task": "transcribe",
        "language": getattr(info, "language", language),
        "duration": duration,
        "text": full_text,
        "segments": out,
        "x_processing_sec": round(elapsed, 2),
        "x_device": _whisper_info["device"],
        "x_model": WHISPER_MODEL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
