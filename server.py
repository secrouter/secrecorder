# Copyright 2026 Austin Probe
# SPDX-License-Identifier: Apache-2.0
"""Whisper ASR server with an OpenAI-compatible API — multi-backend.

Serves ``POST /v1/audio/transcriptions`` (multipart, ``response_format=verbose_json``,
``timestamp_granularities[]=word``) and returns a top-level ``words[]`` array with real word-level
timestamps — the OpenAI verbose_json shape, so existing OpenAI clients work by pointing ``base_url``
here. Also serves the OpenAI Models API (``/v1/models``) and a ``/health`` liveness check.

**Pluggable ASR backend** (``WHISPER_BACKEND=auto|mlx|faster-whisper``; auto picks whichever is
installed):
  * ``mlx``            — Apple MLX (Metal) on Apple Silicon.
  * ``faster-whisper`` — CTranslate2 on NVIDIA CUDA (or CPU). ``WHISPER_DEVICE=auto|cuda|cpu``,
                         ``WHISPER_COMPUTE_TYPE`` (default float16 on cuda, int8 on cpu).
Both normalise to the same segments/words shape, so the endpoints, reliability, and diarization
below are backend-agnostic.

Optional **speaker diarization** (pyannote ``community-1``, PyTorch — CUDA/MPS/CPU): pass
``diarize=true`` (or set ``WHISPER_DIARIZE=1`` to default it on) and every word/segment gains a
``"speaker"`` (``SPEAKER_00``…) plus a top-level ``speakers`` talk-time summary. Extra fields are
ignored by clients that don't know them, so the contract stays backward compatible. Diarization
failures degrade gracefully: the transcription is returned with a ``diarization_error`` instead of
a 500. Requires ``HF_TOKEN`` (gated model) — read from the environment or ``.env`` next to this file.

Optional **speaker recognition** (``speakers.py`` — stdlib SQLite, no extra model): enroll named
voiceprints (``POST /v1/speakers`` from a vector, or ``/v1/speakers/from-audio`` from a sample) and
pass ``identify=true`` so a recording's diarized speakers are matched to them by cosine similarity —
each ``speakers`` entry then also carries ``name`` + ``match_score``. ``identify=true`` implies
diarization; the library persists to ``SPEAKER_DB``.

Reliability on large recordings:
  * the blocking transcription runs in a worker thread (``asyncio.to_thread``), so the event loop
    — and ``/health`` — stays responsive for the whole multi-minute job instead of hanging;
  * GPU work is serialized by a semaphore (``WHISPER_MAX_CONCURRENCY``, default 1) so concurrent
    large recordings queue instead of contending for / exhausting GPU memory;
  * the upload is streamed to disk in chunks (bounded memory), with an optional size cap;
  * the backend's GPU cache is released after each job to curb cross-job memory growth.

    WHISPER_BACKEND=faster-whisper WHISPER_MODEL=large-v3 WHISPER_DEVICE=cuda \\
        uv run uvicorn server:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
# pyannote runs on torch; a few ops still lack MPS kernels on Apple — fall back per-op.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from speakers import SpeakerLibrary, SpeakerError
import audit
import auth
from summarize import summarize as run_summary, SummarizeError, enabled as summarize_enabled, status as summarize_status

BACKEND = os.environ.get("WHISPER_BACKEND", "auto").strip().lower()
MODEL = os.environ.get("WHISPER_MODEL", "").strip()  # empty → per-backend default below
# faster-whisper device / precision (ignored by mlx).
DEVICE = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "").strip()
# Serialize GPU work: one transcription at a time by default, so concurrent large recordings queue
# instead of contending for (and exhausting) GPU memory. Raise only if the GPU can hold N at once.
MAX_CONCURRENCY = max(1, int(os.environ.get("WHISPER_MAX_CONCURRENCY", "1")))
# Optional upload cap in MiB; 0 = unlimited. Guards the disk against a runaway upload.
MAX_UPLOAD_MB = int(os.environ.get("WHISPER_MAX_UPLOAD_MB", "0"))

# Dead-air guard. Whisper HALLUCINATES on silence — an empty/near-silent recording (e.g. a voice
# memo where the mic never captured anything) comes back as fabricated junk ("Thank you.", repeated
# phrases, ...). Before ASR we measure the clip's PEAK volume (ffmpeg volumedetect); if even the
# loudest moment is below this many dBFS, we return an empty transcript instead of running whisper.
# Real speech peaks well above -45 dB; a silent clip reads <= -60 dB (or -inf). Set >= 0 to disable.
SILENCE_MAX_DB = float(os.environ.get("WHISPER_SILENCE_MAX_DB", "-45"))
# Diarization runs on a SEPARATE semaphore (torch/MPS) from ASR (MLX), so episode N's diarization
# doesn't block episode N+1's transcription — the two frameworks pipeline on the GPU.
DIA_CONCURRENCY = max(1, int(os.environ.get("WHISPER_DIA_CONCURRENCY", "1")))
# Prewarm: load the model (and optionally the diarizer) at startup rather than on the first
# request, so a restart has no cold-start hit. PREWARM_DIARIZER covers per-request diarize workloads
# (where WHISPER_DIARIZE default-off but the client still sends diarize=true).
PREWARM = os.environ.get("WHISPER_PREWARM", "0").strip().lower() in ("1", "true", "yes", "on")
PREWARM_DIARIZER = os.environ.get("WHISPER_PREWARM_DIARIZER", "0").strip().lower() in ("1", "true", "yes", "on")

# Speaker diarization (opt-in per request via diarize=true; WHISPER_DIARIZE=1 defaults it on).
DIARIZE_MODEL = os.environ.get("WHISPER_DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
DIARIZE_DEFAULT = os.environ.get("WHISPER_DIARIZE", "0").lower() in ("1", "true", "yes", "on")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
# Hard on/off for diarization. On hardware where pyannote/torch can't load — an old GPU (torch has
# no kernels for it) or a missing system FFmpeg (torchcodec) — the model load can ABORT natively,
# which a Python try/except can't catch. Set WHISPER_ALLOW_DIARIZE=0 there so a diarize=true request
# is ignored (returns a normal transcript) instead of crashing the process.
DIARIZE_ENABLED = os.environ.get("WHISPER_ALLOW_DIARIZE", "1").strip().lower() in ("1", "true", "yes", "on")

# Speaker recognition: a persistent SQLite library of named voiceprints (speakers.py). Enrolled
# speakers are matched against a recording's diarized voiceprints by cosine similarity, so the same
# person is recognized across recordings. The store itself needs no model (enroll from a vector);
# identify + enroll-from-audio require diarization. SPEAKER_LIBRARY=0 disables the feature entirely
# (endpoints 404, identify ignored) for a locked-down / read-only deployment.
SPEAKER_LIBRARY_ENABLED = os.environ.get("SPEAKER_LIBRARY", "1").strip().lower() in ("1", "true", "yes", "on")
SPEAKER_DB = os.environ.get("SPEAKER_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "speakers.db"))
SPEAKER_IDENTIFY_DEFAULT = os.environ.get("SPEAKER_IDENTIFY", "0").strip().lower() in ("1", "true", "yes", "on")
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_IDENTIFY_THRESHOLD", "0.5"))


def _resolve_backend_name() -> str:
    """Which backend to use — resolved cheaply (no model load) so /health and /v1/models work
    before the first transcription. ``auto`` prefers mlx when importable, else faster-whisper."""
    if BACKEND in ("mlx", "faster-whisper", "faster_whisper", "ctranslate2"):
        return "mlx" if BACKEND == "mlx" else "faster-whisper"
    return "mlx" if importlib.util.find_spec("mlx_whisper") else "faster-whisper"


BACKEND_NAME = _resolve_backend_name()
# Per-backend default — both large-v3-turbo: the MLX build on Apple, a CTranslate2 turbo build on
# faster-whisper (CUDA/CPU). Pinned to the exact CT2 repo (not the version-specific alias). Override
# either with WHISPER_MODEL.
_DEFAULT_MODEL = {"mlx": "mlx-community/whisper-large-v3-turbo"}.get(
    BACKEND_NAME, "deepdml/faster-whisper-large-v3-turbo-ct2")
MODEL_ID = MODEL or _DEFAULT_MODEL  # display id; the heavy model load stays lazy in the backend


# --- ASR backend abstraction --------------------------------------------------------------------
# A backend loads a model once and turns an audio path into a normalised result:
#   {"language": str, "text": str,
#    "segments": [{"start": s, "end": s, "text": str, "words": [{"word","start","end"}]}]}


class MlxBackend:
    name = "mlx"

    def __init__(self, model_id: str) -> None:
        import mlx.core as mx  # noqa: F401 — validate the import + hold for clear_cache
        self.model_id = model_id
        self.device = "mps"
        self._mx = mx

    def transcribe(self, path: str, initial_prompt: str | None = None) -> dict:
        import mlx_whisper
        # condition_on_previous_text=False: the default lets an uncertain/garbled segment bias
        # every later segment in the same call, which is how Whisper gets stuck repeating a
        # word or phrase — most visible on live's short, rapidly re-transcribed windows.
        return mlx_whisper.transcribe(path, path_or_hf_repo=self.model_id, word_timestamps=True,
                                      initial_prompt=initial_prompt, condition_on_previous_text=False)

    def clear_cache(self) -> None:
        for fn in (getattr(self._mx, "clear_cache", None),
                   getattr(getattr(self._mx, "metal", None), "clear_cache", None)):
            if callable(fn):
                fn()
                return


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, model_id: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        compute_type = compute_type or ("float16" if device == "cuda" else "int8")
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self._model = WhisperModel(model_id, device=device, compute_type=compute_type)

    def transcribe(self, path: str, initial_prompt: str | None = None) -> dict:
        # condition_on_previous_text=False: see MlxBackend.transcribe — same repetition-loop risk.
        segments, info = self._model.transcribe(path, word_timestamps=True, initial_prompt=initial_prompt,
                                                 condition_on_previous_text=False)
        out = []
        for s in segments:  # generator — consuming it runs the transcription
            out.append({
                "start": float(s.start or 0.0), "end": float(s.end or 0.0), "text": s.text or "",
                "words": [{"word": w.word, "start": float(w.start or 0.0), "end": float(w.end or 0.0)}
                          for w in (s.words or [])],
            })
        return {"language": info.language, "text": " ".join(x["text"] for x in out), "segments": out}

    def clear_cache(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort
            pass


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch may be absent; ask ctranslate2 directly
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:  # noqa: BLE001
            return False


# Separate semaphores so ASR (MLX) and diarization (torch/MPS) PIPELINE instead of serializing:
# a transcription can run while a previous episode's diarization is still going. Both bind to the
# running loop on first await (py3.11+).
_asr_sem = asyncio.Semaphore(MAX_CONCURRENCY)   # MLX transcription
_dia_sem = asyncio.Semaphore(DIA_CONCURRENCY)   # pyannote diarization
_inflight = 0  # jobs queued or running (single-threaded loop → += is safe)

_backend = None
_backend_lock = threading.Lock()


def _prewarm() -> None:
    """Load the model — and the diarizer if PREWARM_DIARIZER — once, off the request path. Runs in
    a background thread so the server accepts connections immediately; the load-locks in
    ``_get_backend``/``_load_diarizer`` make a racing first request wait rather than double-load."""
    try:
        _get_backend()
    except Exception as e:  # noqa: BLE001 — a prewarm failure must not stop the server booting
        print(f"[whisper] backend prewarm failed (will retry lazily): {e}")
    if PREWARM_DIARIZER and DIARIZE_ENABLED:
        try:
            _load_diarizer()
        except Exception as e:  # noqa: BLE001
            print(f"[whisper] diarizer prewarm failed (will retry lazily): {e}")


@contextlib.asynccontextmanager
async def _lifespan(_app):
    if PREWARM or PREWARM_DIARIZER:
        threading.Thread(target=_prewarm, name="prewarm", daemon=True).start()
    yield


app = FastAPI(title="SecRecorder", version="0.7.0", lifespan=_lifespan)

# Built-in web UI (record / upload -> transcribe), served same-origin as the API.
# Read once at startup from ui.html next to this file; optional (API works without it).
_UI_HTML = ""
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"), encoding="utf-8") as _f:
        _UI_HTML = _f.read()
except OSError:
    pass

# Optional SSO auth (auth.py): mounts /auth/* + an enforcement middleware that guards /v1/* and
# bounces the UI to SSO login. A no-op for request handling unless SECRECORDER_OIDC_* is configured.
auth.install(app)

_diarizer = None  # lazy-loaded pyannote pipeline (first diarize request pays the load)
_diarizer_device = ""
_diarizer_lock = threading.Lock()


def _get_backend():
    """Load the ASR backend + model once (thread-safe; first transcription pays the load)."""
    global _backend
    with _backend_lock:
        if _backend is not None:
            return _backend
        if BACKEND_NAME == "mlx":
            _backend = MlxBackend(MODEL_ID)
        else:
            _backend = FasterWhisperBackend(MODEL_ID, DEVICE, COMPUTE_TYPE)
        print(f"[whisper] backend={_backend.name} model={_backend.model_id} device={_backend.device}")
        return _backend


def _load_diarizer():
    """Load the pyannote pipeline once (thread-safe), preferring CUDA, then MPS, then CPU."""
    global _diarizer, _diarizer_device
    with _diarizer_lock:
        if _diarizer is not None:
            return _diarizer
        import torch
        from pyannote.audio import Pipeline

        pipe = Pipeline.from_pretrained(DIARIZE_MODEL, token=HF_TOKEN)
        if pipe is None:  # gated model not accepted / bad token
            raise RuntimeError(
                f"could not load {DIARIZE_MODEL} — accept its conditions on HuggingFace "
                "and set HF_TOKEN")
        _diarizer_device = "cpu"
        for dev in ("cuda", "mps"):
            avail = (dev == "cuda" and torch.cuda.is_available()) or \
                    (dev == "mps" and torch.backends.mps.is_available())
            if avail:
                try:
                    pipe.to(torch.device(dev))
                    _diarizer_device = dev
                    break
                except Exception:  # noqa: BLE001 — device move failed; try the next / CPU
                    pass
        _diarizer = pipe
        print(f"[whisper] diarizer loaded: {DIARIZE_MODEL} on {_diarizer_device}")
        return _diarizer


_library = None  # lazy-opened speaker library (SQLite); no model load
_library_lock = threading.Lock()


def _get_library() -> SpeakerLibrary:
    """Open the speaker library (SQLite at SPEAKER_DB) once, thread-safe. Cheap — no model load."""
    global _library
    with _library_lock:
        if _library is None:
            _library = SpeakerLibrary(SPEAKER_DB)
        return _library


def _require_library() -> SpeakerLibrary:
    """The library, or a 404 when the feature is disabled for this host (SPEAKER_LIBRARY=0)."""
    if not SPEAKER_LIBRARY_ENABLED:
        raise HTTPException(status_code=404, detail="speaker library is disabled on this host")
    return _get_library()


def _principal_sub(request: Request) -> str:
    """The authenticated caller's `sub` for audit attribution, or "anonymous" — auth is off by
    default (open service), and even when it's on some routes may allow an anonymous caller."""
    p = auth.current_principal(request)
    return p.sub if p else "anonymous"


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _json_safe(obj, path: str = "", found: list | None = None):
    """Replace non-finite floats (NaN / ±inf) with None, recursively.

    Starlette's JSONResponse serializes with ``allow_nan=False``, so ONE NaN anywhere in the
    payload raises and turns the whole request into a 500 — losing an entire episode's transcript.
    mlx-whisper emits NaN in segment scores (``avg_logprob`` / ``compression_ratio`` /
    ``no_speech_prob``) on degenerate audio (silence, music beds), and we pass ``segments`` through
    verbatim. Null is the honest JSON for "no value" and every consumer already tolerates a missing
    score; the words/timings the client actually needs are unaffected."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if found is not None:
            found.append(path or "<root>")
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v, f"{path}.{k}" if path else str(k), found) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v, f"{path}[]", found) for v in obj]
    return obj


def _speaker_embeddings(out) -> dict[str, list[float]]:
    """Per-speaker voiceprints from a pyannote 4.x ``DiarizeOutput`` → ``{label: vector}``.

    These let a CHUNKING client link speakers across independently-diarized chunks (chunk 2's
    SPEAKER_00 is not chunk 1's) by cosine similarity — measured ~0.92-0.98 for the same voice vs
    <=0.30 for different ones. Rows are ordered by ``speaker_diarization.labels()``. A speaker with
    too little speech can yield a non-finite vector; those are omitted so the client treats them as
    unmatched (a new speaker) rather than matching on garbage."""
    raw = getattr(out, "speaker_embeddings", None)
    base = getattr(out, "speaker_diarization", None)
    if raw is None or base is None:
        return {}
    emb: dict[str, list[float]] = {}
    for i, lbl in enumerate(base.labels()):
        if i >= len(raw):
            break
        vec = [float(x) for x in raw[i]]
        if vec and all(math.isfinite(x) for x in vec):
            emb[str(lbl)] = [round(x, 6) for x in vec]
    return emb


_FFMPEG = shutil.which("ffmpeg")


def _peak_volume_db(path: str) -> float | None:
    """The clip's PEAK volume in dBFS via ffmpeg's ``volumedetect`` (~0 = full scale; silence reads
    very low). Returns None if ffmpeg is absent or the probe fails — the caller then can't tell and
    transcribes normally (fail-open, never drop real audio)."""
    if not _FFMPEG:
        return None
    try:
        proc = subprocess.run(
            [_FFMPEG, "-nostdin", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (proc.stderr or "").splitlines():
        if "max_volume:" in line:  # e.g. "[Parsed_volumedetect_0 @ ..] max_volume: -12.3 dB"
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


def _to_wav16k(path: str):
    """Transcode any input to 16 kHz mono PCM WAV via ffmpeg. pyannote decodes a file in chunks and
    its strict crop raises on compressed input — AAC/m4a encoder priming makes a chunk come back a
    few samples short ("…477888 samples instead of the expected 480000"). PCM WAV has exact sample
    counts, so diarization is format-agnostic. Returns the new path, or None if ffmpeg is absent
    (the caller then falls back to the original file)."""
    if not _FFMPEG:
        return None
    out = tempfile.NamedTemporaryFile(suffix=".dia.wav", delete=False).name
    try:
        subprocess.run([_FFMPEG, "-nostdin", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le", "-f", "wav", out], check=True, capture_output=True)
        return out
    except Exception:  # noqa: BLE001 — fall back to the raw upload path
        try:
            os.unlink(out)
        except OSError:
            pass
        return None


def _diarize(path: str) -> tuple[list[tuple[float, float, str]], dict[str, list[float]]]:
    """Run diarization → (sorted ``[(start_s, end_s, label)]`` turns, ``{label: embedding}``).
    The upload is first transcoded to 16 kHz mono PCM WAV (``_to_wav16k``) so pyannote's per-chunk
    cropping doesn't choke on compressed formats (m4a/AAC).
    pyannote 4.x returns a ``DiarizeOutput`` — use its ``exclusive_speaker_diarization``
    (non-overlapping turns, made for transcription alignment); older pipelines return the
    ``Annotation`` directly (and carry no embeddings)."""
    wav = _to_wav16k(path)
    try:
        out = _load_diarizer()(wav or path)
    finally:
        if wav:
            try:
                os.unlink(wav)
            except OSError:
                pass
    annotation = (getattr(out, "exclusive_speaker_diarization", None)
                  or getattr(out, "speaker_diarization", None) or out)
    turns = [(float(seg.start), float(seg.end), str(label))
             for seg, _track, label in annotation.itertracks(yield_label=True)]
    turns.sort()
    return turns, _speaker_embeddings(out)


def _assign_speakers(words: list[dict], turns: list[tuple[float, float, str]]) -> None:
    """Stamp each word dict with the speaker whose turn contains its midpoint (nearest turn when
    the word falls in a gap). In place; no-op without turns."""
    if not turns:
        return
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        best, best_d = None, None
        for ts, te, label in turns:
            if ts <= mid < te:
                best = label
                break
            d = (ts - mid) if mid < ts else (mid - te)
            if best_d is None or d < best_d:
                best, best_d = label, d
        w["speaker"] = best


def _transcribe_only(path: str, initial_prompt: str | None = None) -> dict:
    """Blocking transcription — ALWAYS called via ``asyncio.to_thread`` so it never blocks the event
    loop. ``initial_prompt`` biases/continues decoding (OpenAI ``prompt``), used by live mode to
    carry the last committed text across rolling windows. Releases the GPU cache afterwards."""
    backend = _get_backend()
    try:
        return backend.transcribe(path, initial_prompt=initial_prompt)
    finally:
        backend.clear_cache()


def _diarize_safe(path: str) -> tuple[list, dict, str]:
    """Blocking diarization — ALWAYS via ``asyncio.to_thread``. Returns (turns, embeddings, error);
    a failure degrades to no speakers (error string) instead of failing the transcription job."""
    try:
        turns, emb = _diarize(path)
        return turns, emb, ""
    except Exception as e:  # noqa: BLE001 — transcript is still good without speakers
        return [], {}, f"{type(e).__name__}: {e}"


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    """Built-in web UI: record from the mic or upload a file, then transcribe (with optional
    speaker labels). Same-origin as the API, so no CORS. Disabled if ui.html is missing."""
    if not _UI_HTML:
        return HTMLResponse("<h1>SecRecorder</h1><p>UI unavailable (ui.html not found next to server.py). The API is at <code>/v1/audio/transcriptions</code>.</p>")
    return HTMLResponse(_UI_HTML)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": BACKEND_NAME, "model": MODEL_ID,
            "max_concurrency": MAX_CONCURRENCY, "dia_concurrency": DIA_CONCURRENCY,
            "in_flight": _inflight, "prewarm": PREWARM or PREWARM_DIARIZER,
            "diarize_enabled": DIARIZE_ENABLED,
            "diarize_default": DIARIZE_DEFAULT, "diarize_model": DIARIZE_MODEL,
            "diarizer_loaded": _diarizer is not None,
            "loaded": _backend is not None,
            **({"device": _backend.device} if _backend is not None else {}),
            **({"diarizer_device": _diarizer_device} if _diarizer is not None else {}),
            "ffmpeg": _FFMPEG is not None,
            "speaker_library_enabled": SPEAKER_LIBRARY_ENABLED,
            "identify_default": SPEAKER_IDENTIFY_DEFAULT,
            "identify_threshold": SPEAKER_THRESHOLD,
            **({"speaker_count": _library.count()} if _library is not None else {}),
            "auth": auth.status(), "summarize": summarize_status(), "audit": audit.status()}


# --- OpenAI-compatible Models API -------------------------------------------------
_CREATED = 1704067200  # stable "created" timestamp (2024-01-01) — this server serves one model


def _model_obj(model_id: str) -> dict:
    owner = model_id.split("/", 1)[0] if "/" in model_id else BACKEND_NAME
    return {"id": model_id, "object": "model", "created": _CREATED, "owned_by": owner}


@app.get("/v1/models")
def list_models() -> dict:
    """OpenAI GET /v1/models — the one model this server has loaded."""
    return {"object": "list", "data": [_model_obj(MODEL_ID)]}


@app.get("/v1/models/{model_id:path}")
def retrieve_model(model_id: str):
    """OpenAI GET /v1/models/{id}. An empty id (a trailing-slash /v1/models/) lists; an id that
    isn't the loaded model returns an OpenAI-shaped 404."""
    if not model_id:
        return list_models()
    if model_id == MODEL_ID:
        return _model_obj(MODEL_ID)
    return JSONResponse(status_code=404, content={"error": {
        "message": f"The model '{model_id}' does not exist",
        "type": "invalid_request_error", "param": "model", "code": "model_not_found"}})


# --- Speaker library: enroll named voiceprints, recognize them across recordings ---------------
# A voiceprint is a `speakers[].embedding` from a diarized transcription. Enroll one under a name,
# then pass identify=true on a later transcription to have its speakers matched back to that name.


class EnrollBody(BaseModel):
    name: str
    embedding: list[float]
    source: str | None = None
    meta: dict | None = None


class SampleBody(BaseModel):
    embedding: list[float]
    source: str | None = None


@app.get("/v1/speakers")
def speakers_list() -> dict:
    """List enrolled speakers (id, name, sample count, timestamps) — newest-updated first."""
    return {"object": "list", "data": _require_library().list_speakers()}


@app.post("/v1/speakers", status_code=201)
def speakers_enroll(request: Request, body: EnrollBody) -> dict:
    """Enroll a speaker from a voiceprint vector — e.g. a ``speakers[].embedding`` returned by a
    prior transcription. To enroll straight from an audio sample use ``/v1/speakers/from-audio``."""
    try:
        spk = _require_library().enroll(body.name, body.embedding, source=body.source, meta=body.meta)
    except SpeakerError as e:
        audit.record("speaker.enroll", principal=_principal_sub(request), source_ip=_client_ip(request),
                     outcome="error", detail={"source": body.source or "vector", "error": str(e)})
        raise HTTPException(status_code=400, detail=str(e)) from e
    # target = the speaker id (Spec B.1); detail is metadata only — never the enrolled name/voice
    # (voiceprints and the identity they carry are exactly what B.3 says never lands in the log).
    audit.record("speaker.enroll", principal=_principal_sub(request), source_ip=_client_ip(request),
                 target={"speakerId": spk["id"]}, detail={"source": body.source or "vector"})
    return spk


@app.get("/v1/speakers/{speaker_id}")
def speakers_get(speaker_id: str, centroid: bool = False) -> dict:
    """One speaker. ``?centroid=1`` includes its mean voiceprint."""
    spk = _require_library().get(speaker_id, with_centroid=centroid)
    if spk is None:
        raise HTTPException(status_code=404, detail=f"unknown speaker: {speaker_id}")
    return spk


@app.post("/v1/speakers/{speaker_id}/samples", status_code=201)
def speakers_add_sample(request: Request, speaker_id: str, body: SampleBody) -> dict:
    """Add another voiceprint to an existing speaker — more samples sharpen recognition."""
    try:
        spk = _require_library().add_sample(speaker_id, body.embedding, source=body.source)
    except SpeakerError as e:
        code = 404 if "unknown speaker" in str(e) else 400
        audit.record("speaker.update", principal=_principal_sub(request), source_ip=_client_ip(request),
                     target={"speakerId": speaker_id}, outcome="error", detail={"error": str(e)})
        raise HTTPException(status_code=code, detail=str(e)) from e
    audit.record("speaker.update", principal=_principal_sub(request), source_ip=_client_ip(request),
                 target={"speakerId": speaker_id}, detail={"source": body.source or "vector",
                                                            "samples": spk.get("samples")})
    return spk


@app.delete("/v1/speakers/{speaker_id}")
def speakers_delete(request: Request, speaker_id: str) -> dict:
    """Remove a speaker and all its voiceprints."""
    if not _require_library().delete(speaker_id):
        raise HTTPException(status_code=404, detail=f"unknown speaker: {speaker_id}")
    audit.record("speaker.delete", principal=_principal_sub(request), source_ip=_client_ip(request),
                 target={"speakerId": speaker_id})
    return {"deleted": speaker_id}


@app.post("/v1/speakers/from-audio", status_code=201)
async def speakers_enroll_audio(request: Request, file: UploadFile = File(...), name: str = Form(...)) -> dict:
    """Enroll a speaker from an audio sample: diarize it, take the dominant speaker's voiceprint,
    and store it under ``name``. Use a clean, single-speaker sample. Requires diarization."""
    lib = _require_library()
    if not DIARIZE_ENABLED:
        raise HTTPException(status_code=503, detail="diarization disabled; enroll from a vector instead")
    path, size = await _spool_upload(file)
    if size == 0:
        os.unlink(path)
        raise HTTPException(status_code=400, detail="empty file")
    try:
        async with _dia_sem:
            turns, emb, err = await asyncio.to_thread(_diarize_safe, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if err:
        raise HTTPException(status_code=422, detail=f"diarization failed: {err}")
    if not emb:
        raise HTTPException(status_code=422, detail="no usable voiceprint in the sample")
    # Dominant speaker = most talk time among labels that produced a usable voiceprint.
    talk: dict[str, float] = {}
    for ts, te, label in turns:
        talk[label] = talk.get(label, 0.0) + (te - ts)
    dominant = max(emb, key=lambda lbl: talk.get(lbl, 0.0))
    try:
        spk = lib.enroll(name, emb[dominant], source=f"audio:{os.path.basename(file.filename or 'sample')}")
    except SpeakerError as e:
        audit.record("speaker.enroll", principal=_principal_sub(request), source_ip=_client_ip(request),
                     outcome="error", detail={"source": "audio", "error": str(e)})
        raise HTTPException(status_code=400, detail=str(e)) from e
    audit.record("speaker.enroll", principal=_principal_sub(request), source_ip=_client_ip(request),
                 target={"speakerId": spk["id"]}, detail={"source": "audio", "speakersInSample": len(talk)})
    return {**spk, "speakers_in_sample": len(talk)}


async def _spool_upload(file: UploadFile) -> tuple[str, int]:
    """Stream the multipart upload to a temp ``.wav`` in 1 MiB chunks (bounded memory — never
    holds the whole recording in RAM). Returns (path, size_bytes). Enforces
    ``WHISPER_MAX_UPLOAD_MB`` if set (413)."""
    cap = MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
        while True:
            chunk = await file.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            size += len(chunk)
            if cap and size > cap:
                tmp.close()
                os.unlink(path)
                raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_MB} MiB")
            tmp.write(chunk)
    return path, size


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request,
                         file: UploadFile = File(...),
                         diarize: str | None = Form(None),
                         identify: str | None = Form(None),
                         prompt: str | None = Form(None),
                         summarize: str | None = Form(None)) -> JSONResponse:
    """OpenAI-compatible transcription. The ``model`` / ``response_format`` /
    ``timestamp_granularities[]`` form fields the client sends are accepted and ignored — this
    server always uses the loaded model and always returns verbose_json + word[].
    ``diarize=true`` adds ``speaker`` to every word/segment + a ``speakers`` summary."""
    global _inflight
    # No recording is persisted anywhere in this server (each upload is transcribed and discarded —
    # see _spool_upload/finally below), so there is no durable "recording id" to audit against. A
    # fresh per-request id is the closest equivalent (Spec B.1 "target: recording id"), letting an
    # operator correlate this one audit entry with server logs / a client-side request id.
    audit_rid = uuid.uuid4().hex[:12]
    audit_principal = _principal_sub(request)
    audit_ip = _client_ip(request)
    requested_diarize = (DIARIZE_DEFAULT if diarize is None
                         else diarize.strip().lower() in ("1", "true", "yes", "on"))
    requested_identify = (SPEAKER_IDENTIFY_DEFAULT if identify is None
                          else identify.strip().lower() in ("1", "true", "yes", "on"))
    want_identify = requested_identify and SPEAKER_LIBRARY_ENABLED
    # Honour the hard kill-switch: if diarization is disabled for this host, ignore the request
    # (return a plain transcript) rather than attempt a load that could natively abort.
    # Identification needs the per-speaker voiceprints, so asking to identify turns diarization on.
    want_diarize = (requested_diarize or want_identify) and DIARIZE_ENABLED
    path, size = await _spool_upload(file)
    if size == 0:
        os.unlink(path)
        raise HTTPException(status_code=400, detail="empty file")

    # Dead-air guard (SILENCE_MAX_DB): a silent clip is returned as an EMPTY transcript rather than
    # letting whisper hallucinate junk on it. Fail-open — a probe that can't measure the level
    # (ffmpeg missing/errored → None) transcribes normally.
    if SILENCE_MAX_DB < 0:
        peak = await asyncio.to_thread(_peak_volume_db, path)
        if peak is not None and peak < SILENCE_MAX_DB:
            os.unlink(path)
            audit.record("transcription.run", principal=audit_principal, source_ip=audit_ip,
                         target={"requestId": audit_rid},
                         detail={"sizeBytes": size, "silenceGated": True, "durationSec": 0.0,
                                 "diarize": requested_diarize, "identify": requested_identify,
                                 "maxVolumeDb": peak})
            return JSONResponse({
                "task": "transcribe",
                "language": "en",
                "duration": 0.0,
                "text": "",
                "words": [],
                "segments": [],
                "speakers": [],
                "silence": True,
                "max_volume_db": peak,
            })

    _inflight += 1
    t0 = time.monotonic()
    turns: list = []
    dia_emb: dict = {}
    dia_err = ""
    try:
        async with _asr_sem:  # MLX transcription; extra requests queue here, the loop stays free
            result = await asyncio.to_thread(_transcribe_only, path, prompt)
        if want_diarize:
            # Separate semaphore/GPU queue: this diarization overlaps the NEXT episode's
            # transcription instead of blocking it.
            async with _dia_sem:
                turns, dia_emb, dia_err = await asyncio.to_thread(_diarize_safe, path)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 — surface the reason instead of a bare 500
        audit.record("transcription.run", principal=audit_principal, source_ip=audit_ip,
                     target={"requestId": audit_rid}, outcome="error",
                     detail={"sizeBytes": size, "diarize": requested_diarize,
                             "identify": requested_identify, "error": f"{type(e).__name__}: {e}"})
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}") from e
    finally:
        _inflight -= 1
        try:
            os.unlink(path)
        except OSError:
            pass
    elapsed = time.monotonic() - t0

    segments = result.get("segments", []) or []
    # Flatten segment words → the top-level words[] most clients read (real per-word timings).
    words = [
        {"word": w.get("word", ""), "start": float(w.get("start", 0.0)), "end": float(w.get("end", 0.0))}
        for seg in segments for w in (seg.get("words") or [])
    ]
    payload: dict = {
        "task": "transcribe",
        "language": result.get("language", "en"),
        "duration": float(segments[-1].get("end", 0.0)) if segments else 0.0,
        "text": result.get("text", ""),
        "words": words,        # the primary field clients read (word-level timestamps)
        "segments": segments,  # kept for the parser's segment fallback + general compatibility
    }
    n_speakers = 0
    if want_diarize and turns:
        _assign_speakers(words, turns)
        for seg in segments:  # segment speaker = the turn containing its midpoint
            mids = [{"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0))}]
            _assign_speakers(mids, turns)
            seg["speaker"] = mids[0].get("speaker")
        talk: dict[str, float] = {}
        for ts, te, label in turns:
            talk[label] = talk.get(label, 0.0) + (te - ts)
        # `embedding` lets a chunking client link this chunk's speakers to the previous chunks'
        # (see _speaker_embeddings). Omitted for a speaker whose vector wasn't usable.
        payload["speakers"] = [{"id": s, "talk_time": round(t, 2),
                                **({"embedding": dia_emb[s]} if s in dia_emb else {})}
                               for s, t in sorted(talk.items())]
        n_speakers = len(talk)
        if want_identify and dia_emb:
            # Match each diarized voiceprint to an enrolled speaker. Off the loop (cheap, but a
            # large library shouldn't stall it). Annotate the summary; labels stay stable.
            matches = await asyncio.to_thread(_get_library().identify_many, dia_emb, SPEAKER_THRESHOLD)
            for sp in payload["speakers"]:
                mt = matches.get(sp["id"])
                if mt:
                    sp["name"] = mt["name"]
                    sp["speaker_id"] = mt["speaker_id"]
                    sp["match_score"] = mt["score"]
    elif want_diarize and dia_err:
        payload["diarization_error"] = dia_err
    elif requested_diarize and not DIARIZE_ENABLED:
        payload["diarization_disabled"] = True  # asked for, but disabled on this host
    if requested_identify and not SPEAKER_LIBRARY_ENABLED:
        payload["identification_disabled"] = True  # asked for, but the library is disabled here

    # Optional summarization (summarize.py): a governed LLM call attributed to the authenticated
    # caller (X-Sec-Acting-User → SecRouter). Off unless configured AND requested; a failure
    # degrades gracefully to a summary_error rather than losing the transcript.
    want_summary = summarize is not None and summarize.strip().lower() in ("1", "true", "yes", "on")
    if want_summary and summarize_enabled and payload.get("text", "").strip():
        principal = auth.current_principal(request)
        text = payload["text"]
        try:
            payload["summary"] = await run_summary(text, principal.sub if principal else None)
            # The LLM call itself is governed/audited at SecRouter (attribution, budget, egress) —
            # this is just the request-level accountability event for SecRecorder's own audit trail.
            audit.record("summarize.request", principal=audit_principal, source_ip=audit_ip,
                         target={"requestId": audit_rid},
                         detail={"governedBy": "secrouter", "transcriptChars": len(text)})
        except SummarizeError as e:
            payload["summary_error"] = str(e)
            audit.record("summarize.request", principal=audit_principal, source_ip=audit_ip,
                         target={"requestId": audit_rid}, outcome="error",
                         detail={"governedBy": "secrouter", "transcriptChars": len(text), "error": str(e)})
    elif want_summary and not summarize_enabled:
        payload["summary_disabled"] = True  # asked for, but summarization isn't configured here

    audit.record("transcription.run", principal=audit_principal, source_ip=audit_ip,
                 target={"requestId": audit_rid},
                 detail={"sizeBytes": size, "durationSec": round(payload["duration"], 2),
                         "elapsedSec": round(elapsed, 2), "diarize": want_diarize,
                         "identify": want_identify, "silenceGated": False,
                         "speakerCount": n_speakers, "backend": BACKEND_NAME, "model": MODEL_ID,
                         "summarizeRequested": want_summary})

    dur = payload["duration"]
    rtf = f"{dur / elapsed:.0f}x realtime" if elapsed > 0 else "n/a"
    n_named = sum(1 for sp in payload.get("speakers", []) if sp.get("name"))
    dia_note = ((f", {n_speakers} speakers" + (f" ({n_named} named)" if n_named else ""))
                if n_speakers
                else (f", diarize FAILED: {dia_err}" if dia_err else ""))
    # Scrub non-finite floats LAST, so nothing above has to care. One NaN would otherwise 500 the
    # whole request (Starlette serializes with allow_nan=False) and lose the episode.
    nan_paths: list = []
    payload = _json_safe(payload, found=nan_paths)
    if nan_paths:
        uniq = sorted({p.split("[]")[-1].lstrip(".") or p for p in nan_paths})
        print(f"[whisper] nulled {len(nan_paths)} non-finite value(s) in {uniq[:4]} "
              f"(would otherwise have 500'd this request)")
    print(f"[whisper] {os.path.basename(file.filename or 'audio')}: {size / 1e6:.1f}MB upload, "
          f"{len(words)} words{dia_note}, {dur:.0f}s audio in {elapsed:.1f}s ({rtf})")
    return JSONResponse(payload)


class SummarizeBody(BaseModel):
    text: str


@app.post("/v1/summarize")
async def summarize_text(request: Request, body: SummarizeBody) -> dict:
    """Summarize arbitrary text (e.g. a prior transcript) via the configured governed LLM endpoint —
    the standalone counterpart to ``summarize=true`` on the transcription route. 503 when
    summarization isn't configured; 502 when the LLM call fails."""
    if not summarize_enabled:
        raise HTTPException(status_code=503, detail="summarization is not configured")
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    principal = auth.current_principal(request)
    try:
        summary = await run_summary(body.text, principal.sub if principal else None)
    except SummarizeError as e:
        audit.record("summarize.request", principal=_principal_sub(request), source_ip=_client_ip(request),
                     outcome="error",
                     detail={"governedBy": "secrouter", "transcriptChars": len(body.text), "error": str(e)})
        raise HTTPException(status_code=502, detail=f"summarization failed: {e}") from e
    audit.record("summarize.request", principal=_principal_sub(request), source_ip=_client_ip(request),
                 detail={"governedBy": "secrouter", "transcriptChars": len(body.text)})
    return {"summary": summary}


# --- Audit + CMMC evidence (Spec B.6/B.7) -------------------------------------------------------
# Both routes live under /v1/, so they get the SAME gate every other API route gets: when auth is
# on, a valid principal is required (401 otherwise) — auth.py's middleware already enforces this for
# anything under /v1/. There is no admin/group concept anywhere else in this service (Principal
# carries `groups` from the OIDC token, but nothing in this codebase checks them), so "any
# authenticated principal" IS the strictest gate available here; inventing a new admin-group check
# for just these two routes would be a new, ungated trust boundary of exactly the kind Spec B.4
# warns against. When auth is off (the default, open-service posture), these stay open like the
# rest of /v1/ — consistent, not a special case.


@app.get("/v1/audit/verify")
def audit_verify() -> dict:
    """Verify the audit hash chain (AU-3.3.8 tamper-evidence)."""
    ok, checked, broken = audit.verify_chain(audit.AUDIT_PATH)
    return {"ok": ok, "checked": checked, **({"brokenAtSeq": broken} if broken is not None else {})}


@app.get("/v1/evidence")
def evidence() -> dict:
    """One-shot CMMC evidence bundle (Spec B.6): sanitized config posture, the audit chain's
    verification result, the last 200 audit records, and a small control self-assessment. Never
    includes a secret/token — only model names, thresholds, paths, and feature flags."""
    ok, checked, broken = audit.verify_chain(audit.AUDIT_PATH)
    return {
        "product": "secrecorder",
        "version": app.version,
        "generatedAt": audit.now_iso(),
        "generatedBy": "secrecorder:/v1/evidence",
        "config": {
            "backend": BACKEND_NAME,
            "model": MODEL_ID,
            "diarizeEnabled": DIARIZE_ENABLED,
            "diarizeDefault": DIARIZE_DEFAULT,
            "diarizeModel": DIARIZE_MODEL,
            "silenceMaxDb": SILENCE_MAX_DB,
            "speakerLibraryEnabled": SPEAKER_LIBRARY_ENABLED,
            "speakerIdentifyDefault": SPEAKER_IDENTIFY_DEFAULT,
            "speakerIdentifyThreshold": SPEAKER_THRESHOLD,
            "authEnabled": auth.auth_enabled,
            "ssoLoginReady": auth.sso_ready,
            "bearerReady": auth.bearer_ready,
            "summarizeEnabled": summarize_enabled,
            "summarizeGovernedBy": "secrouter (or whatever endpoint this deployment configured)",
            "auditEnabled": audit.AUDIT_ENABLED,
            "auditPath": audit.AUDIT_PATH,
        },
        "auditChain": {"ok": ok, "checked": checked, **({"brokenAtSeq": broken} if broken is not None else {})},
        "auditRecent": audit.tail_records(audit.AUDIT_PATH, 200),
        "controls": [
            {"id": "AU-3.3.1", "family": "AU",
             "requirement": "Create and retain system audit records for lifecycle/admin events.",
             "status": "implemented",
             "evidence": "audit.jsonl events: transcription.run, speaker.enroll/update/delete, "
                         "summarize.request, auth.failure — see auditRecent[] above."},
            {"id": "AU-3.3.8", "family": "AU",
             "requirement": "Protect audit information and audit logging tools from unauthorized "
                            "access, modification, and deletion.",
             "status": "implemented",
             "evidence": "SHA-256 hash chain (audit.verify_chain); GET /v1/audit/verify; "
                         "0700/0600 at-rest permissions on the log directory/file."},
            {"id": "IA-3.5.2", "family": "IA",
             "requirement": "Authenticate the identity of users, processes, or devices before "
                            "allowing access.",
             "status": "implemented" if auth.auth_enabled else "not enforced on this deployment",
             "evidence": "auth.py OIDC bearer (JWKS/RS256) + browser-login BFF against SecSSO; "
                         "principal.sub carried into every audit record's `principal` field. "
                         "SecRecorder ships auth OFF BY DEFAULT (open service) — the deployer must "
                         "set SECRECORDER_OIDC_ISSUER/_AUDIENCE (or _CLIENT_ID+secret+session) in "
                         "any deployment handling CUI; see docs/control-validation.md."},
        ],
        "notes": [
            {"topic": "SI (silence-gate)",
             "note": "WHISPER_SILENCE_MAX_DB rejects degenerate/near-silent audio before ASR "
                     "(Whisper hallucinates fabricated text on silence); gated runs are flagged "
                     "detail.silenceGated=true in the transcription.run audit event. Informational "
                     "— not asserted against a specific NIST SP 800-171 control id."},
        ],
    }
