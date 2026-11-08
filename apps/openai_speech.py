"""
VieNeu-TTS — OpenAI-compatible speech API (streaming), CPU or GPU.
==================================================================
Drop-in for ``POST /v1/audio/speech``: any OpenAI SDK / Pipecat / LiveKit /
Vercel AI client works by changing ``base_url``. Audio streams as it is
generated — the first bytes leave ~115 ms after the request on an RTX 3060,
~600 ms on a 6-core CPU (see docs/streaming.vi.md for every number).

    uv run python -m apps.openai_speech                 # http://127.0.0.1:8000
    VIENEU_BACKEND=onnx uv run python -m apps.openai_speech   # force CPU

    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
    with client.audio.speech.with_streaming_response.create(
        model="vieneu-v3-turbo", voice="Mai Anh", input="Xin chào! Đây là chế độ streaming của VieNeu, phát tới đâu nghe tới đó.",
        response_format="pcm",              # 48 kHz s16le mono (see `sample_rate` below)
    ) as r:
        for chunk in r.iter_bytes(4096):
            play(chunk)

Request body (OpenAI fields + a few extras, all optional but ``input``):
    model            any string; reported back as-is
    input            text, chunked internally at ``max_chars`` (256)
    voice            preset name (GET /v1/voices) or one added with POST /v1/voices
    response_format  "pcm" (s16le, no header) | "wav" (header with unknown length,
                     plays as it streams). mp3/opus/aac/flac → 400.
    stream_format    "audio" (default: raw chunked body) | "sse" (Server-Sent
                     Events: {"type":"speech.audio.delta","audio":"<base64>"} ...
                     {"type":"speech.audio.done","usage":{...}})
    speed            accepted for compatibility, ignored (header X-VieNeu-Ignored)
    sample_rate      48000 (native) | 24000 (OpenAI's pcm rate) | 16000 | 8000 —
                     resampled on the fly (soxr), per chunk, no extra latency
    temperature, top_k, top_p, repetition_penalty, max_chars   sampling extras,
                     range-checked (out of range → 400): 0-2 / 1-1024 / (0, 1] / 1-2 / 64-512

POST /v1/voices refuses (409) the name of a built-in voice or alias, so a client
cannot swap out a voice other clients rely on (e.g. the default one).

Concurrency: GPU serves ``VIENEU_MAX_STREAMS`` streams at once (continuous
batching, default 16); CPU serves one (fp32) or two (int8) — the ONNX engine
interleaves them frame by frame but they share the cores.
Extra requests wait up to ``VIENEU_QUEUE_TIMEOUT`` s in a queue of at most
``VIENEU_QUEUE`` entries, then get 429 with Retry-After.

Environment:
    VIENEU_BACKEND=auto|onnx|pytorch   VIENEU_DEVICE=auto|cuda|cpu
    VIENEU_PRECISION=fp32|int8 (CPU)   VIENEU_ONNX_DIR=... (local ONNX export)
    VIENEU_MAX_STREAMS=16 (GPU)        VIENEU_QUEUE=16   VIENEU_QUEUE_TIMEOUT=10
    VIENEU_API_KEY=...  (Bearer auth; unset = open — logged as a warning when
                        HOST is not a loopback address)
    VIENEU_WATERMARK=1                 HOST=127.0.0.1  PORT=8000
                                       (HOST=0.0.0.0 to serve other machines; the
                                       Docker profiles set it)
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import math
import os
import re
import struct
import threading
import time
import uuid
import wave
import weakref
from typing import Any, Iterator, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from vieneu import Vieneu
try:
    from apps import user_voices
except ModuleNotFoundError:  # direct: python apps/openai_speech.py
    import user_voices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vieneu.api")

SAMPLE_RATE = 48_000
MODEL_ID = "vieneu-v3-turbo"
RATES = (48_000, 24_000, 16_000, 8_000)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ── engine + admission ───────────────────────────────────────────────────────

class Engine:
    """The model plus the admission gate (how many streams run at once)."""

    def __init__(self):
        backend = os.environ.get("VIENEU_BACKEND", "auto")
        device = os.environ.get("VIENEU_DEVICE", "auto")
        kw: dict = dict(backend=backend, device=device,
                        precision=os.environ.get("VIENEU_PRECISION", "fp32"),
                        backbone_repo=os.environ.get("VIENEU_BACKBONE_DIR", "pnnbao-ump/VieNeu-TTS-v3-Turbo"),
                        onnx_dir=os.environ.get("VIENEU_ONNX_DIR") or None,
                        codec_dir=os.environ.get("VIENEU_CODEC_DIR") or None,
                        max_streams=_env_int("VIENEU_MAX_STREAMS", 16))
        log.info("⏳ loading VieNeu-TTS v3 Turbo (backend=%s device=%s)", backend, device)
        t = time.perf_counter()
        self.tts = Vieneu(mode="v3turbo", **kw)
        self.backend = self.tts.backend
        # The voices and aliases that exist before any client enrolls one;
        # POST /v1/voices may not replace them.
        self.builtin_voices = frozenset(self.tts._preset_voices) | frozenset(self.tts._voice_aliases)
        loaded_user_voices = user_voices.load_user_voices(self.tts)
        self.watermark = os.environ.get("VIENEU_WATERMARK", "1") != "0"
        # GPU: the scheduler batches every stream. CPU: the ONNX engine interleaves
        # calls frame by frame, but they share the cores — measured on a 6-core
        # desktop, fp32 keeps one stream real-time (RTF 0.58; two → 1.19), int8
        # two (RTF 0.35; two → 0.67). VIENEU_MAX_STREAMS overrides either.
        self.sched = self.tts._get_stream_scheduler() if self.backend == "pytorch" else None
        if self.sched is not None:
            self.max_streams = self.sched.B
        else:
            cpu_default = 2 if kw["precision"] == "int8" else 1
            self.max_streams = _env_int("VIENEU_MAX_STREAMS", cpu_default) if "VIENEU_MAX_STREAMS" in os.environ else cpu_default
        self.max_queue = _env_int("VIENEU_QUEUE", self.max_streams)
        self.queue_timeout = float(os.environ.get("VIENEU_QUEUE_TIMEOUT", "10"))
        self._gate = threading.BoundedSemaphore(self.max_streams)
        self._lock = threading.Lock()
        self._seed_lock = threading.Lock()
        self.active = 0
        self.waiting = 0
        # Warm-up: capture the CUDA graphs / load the ONNX graphs before the
        # first client pays for it.
        for _ in self.tts.infer_stream("Xin chào.", apply_watermark=False):
            pass
        log.info("✅ ready in %.1fs: backend=%s max_streams=%d queue=%d", time.perf_counter() - t,
                 self.backend, self.max_streams, self.max_queue)
        if loaded_user_voices:
            log.info("Loaded %d persisted user voice(s)", len(loaded_user_voices))

    def acquire(self) -> None:
        with self._lock:
            if self.waiting >= self.max_queue:
                raise HTTPException(429, "server busy: queue full", headers={"Retry-After": "1"})
            self.waiting += 1
        try:
            if not self._gate.acquire(timeout=self.queue_timeout):
                raise HTTPException(429, "server busy: timed out waiting for a stream slot",
                                    headers={"Retry-After": "2"})
        finally:
            with self._lock:
                self.waiting -= 1
        with self._lock:
            self.active += 1

    def release(self) -> None:
        with self._lock:
            self.active -= 1
        self._gate.release()

    def stream_error(self) -> Optional[BaseException]:
        """Why the GPU stream scheduler died, or None. A dead scheduler refuses
        every request until the process restarts."""
        return getattr(self.sched, "error", None) if self.sched is not None else None

    def voices(self) -> list:
        out = []
        for name, v in self.tts._preset_voices.items():
            is_user = bool(v.get(user_voices.USER_MARK))
            revision = max(1, int(v.get("revision") or 1))
            item = {"id": name, "name": name, "description": v.get("description", ""),
                    "gender": v.get("gender", ""), "featured": v.get("featured"),
                    "aliases": list(v.get("aliases") or []),
                    "source": "user" if is_user else "builtin",
                    "revision": revision,
                    "synthesis": user_voices.normalize_synthesis(v.get("synthesis")),
                    "delivery": user_voices.normalize_delivery(v.get("delivery"))}
            if is_user:
                try:
                    item["profile"] = user_voices.profile_summary(name, revision)
                except (OSError, ValueError, json.JSONDecodeError):
                    item["profile"] = {
                        "voice_id": name,
                        "revision": revision,
                        "reference": {"available": False},
                    }
            out.append(item)
        return out


class _Slot:
    """One ``Engine`` stream slot, released exactly once: by the stream's own
    ``finally``, or — when the body is never iterated because the client left
    while queued — when the dropped body is garbage-collected."""

    def __init__(self, eng: Engine):
        self._eng = eng
        self._lock = threading.Lock()
        self._held = True

    def release(self) -> None:
        with self._lock:
            held, self._held = self._held, False
        if held:
            self._eng.release()


ENGINE: Optional[Engine] = None


def engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine()
    return ENGINE


# ── auth / errors in OpenAI's shape ───────────────────────────────────────────

def _auth(authorization: Optional[str] = Header(default=None)) -> None:
    key = os.environ.get("VIENEU_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "invalid api key")


def _error(status: int, message: str, code: Optional[str] = None, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error",
                                   "code": code or status}}, status_code=status, headers=headers)


app = FastAPI(title="VieNeu-TTS speech API", version="1.0")


@app.exception_handler(HTTPException)
async def _http_exc(_req: Request, exc: HTTPException):
    return _error(exc.status_code, str(exc.detail), headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def _invalid_request(_req: Request, exc: RequestValidationError):
    # Field and reason only: FastAPI's default echoes the rejected value, and a
    # NaN (which json.loads accepts) cannot be encoded back — the 422 became a 500.
    errs = exc.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", ()) if p not in ("body", "query", "header"))
    msg = first.get("msg", "invalid request")
    return _error(400, f"{field}: {msg}" if field else msg)


@app.on_event("startup")
def _startup() -> None:
    engine()


# ── audio helpers ────────────────────────────────────────────────────────────

def _pcm16(x: np.ndarray) -> bytes:
    return (np.asarray(x, dtype=np.float32) * 32767).clip(-32768, 32767).astype("<i2").tobytes()


def _wav_header(rate: int) -> bytes:
    """A WAV header claiming an unknown (huge) length, so players start at once."""
    h = io.BytesIO()
    with wave.open(h, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.setnframes(0)
    b = bytearray(h.getvalue())
    struct.pack_into("<I", b, 4, 0xFFFFFFFF)      # RIFF size
    struct.pack_into("<I", b, len(b) - 4, 0xFFFFFFFF)   # data size
    return bytes(b)


class _Resampler:
    """48 kHz → ``rate`` chunk by chunk with carried state (soxr), no added latency."""

    def __init__(self, rate: int):
        self.rate = rate
        self._rs = None
        if rate != SAMPLE_RATE:
            import soxr
            self._rs = soxr.ResampleStream(SAMPLE_RATE, rate, 1, dtype="float32", quality="HQ")

    def __call__(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        if self._rs is None:
            return x
        return self._rs.resample_chunk(np.asarray(x, dtype=np.float32), last=last)


# ── /v1/audio/speech ─────────────────────────────────────────────────────────

class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str = Field(min_length=1, max_length=20_000)
    voice: Optional[str] = None
    response_format: str = "wav"
    stream_format: str = "audio"
    speed: Optional[float] = None
    instructions: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    # Bounded so one request cannot break the GPU scheduler every stream shares:
    # e.g. a huge top_k overflowed its int64 slot and NaN reached multinomial.
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, allow_inf_nan=False)
    top_k: Optional[int] = Field(None, ge=1, le=1024)   # 1024 = the codec's vocabulary
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0, allow_inf_nan=False)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0, allow_inf_nan=False)
    max_chars: Optional[int] = Field(None, ge=64, le=512)
    gain_db: Optional[float] = Field(None, ge=-12.0, le=12.0, allow_inf_nan=False)
    seed: Optional[int] = Field(None, ge=0, le=0xFFFFFFFF)


def _synthesis_settings(eng: Engine, req: SpeechRequest) -> dict:
    voice_name = eng.tts.resolve_voice_name(req.voice) if req.voice else None
    voice = eng.tts._preset_voices.get(voice_name) if voice_name else None
    settings = user_voices.normalize_synthesis((voice or {}).get("synthesis"))
    for key in settings:
        override = getattr(req, key)
        if override is not None:
            settings[key] = override
    return settings


def _delivery_settings(eng: Engine, req: SpeechRequest) -> dict:
    voice_name = eng.tts.resolve_voice_name(req.voice) if req.voice else None
    voice = eng.tts._preset_voices.get(voice_name) if voice_name else None
    return user_voices.normalize_delivery((voice or {}).get("delivery"))


def _poetry_lines(text: str) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    pending_stanza_break = False
    for raw in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = raw.strip()
        if not cleaned:
            if lines:
                pending_stanza_break = True
            continue
        if pending_stanza_break and lines:
            previous, _ = lines[-1]
            lines[-1] = (previous, True)
            pending_stanza_break = False
        lines.append((cleaned, False))
    return lines


def _candidate_score(audio: np.ndarray) -> float:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(samples) < SAMPLE_RATE // 5 or not np.all(np.isfinite(samples)):
        return -1e9
    absolute = np.abs(samples)
    clipping = float(np.mean(absolute >= 0.995))
    if clipping > 0.002:
        return -1e9
    frame = max(1, SAMPLE_RATE // 50)
    usable = len(samples) - (len(samples) % frame)
    if usable <= 0:
        return -1e9
    envelope = np.sqrt(
        np.mean(np.square(samples[:usable].reshape(-1, frame)), axis=1) + 1e-12
    )
    voiced = envelope[envelope > 0.003]
    if len(voiced) < 3:
        return -1e9
    dynamics = float(np.std(20.0 * np.log10(voiced + 1e-8)))
    tail_size = min(len(samples), int(SAMPLE_RATE * 0.35))
    tail_rms = float(np.sqrt(np.mean(np.square(samples[-tail_size:])) + 1e-12))
    body_rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    tail_ratio = min(1.5, tail_rms / max(body_rms, 1e-6))
    return dynamics * 0.7 + tail_ratio * 0.3 - clipping * 100.0


def _fade_tail(audio: np.ndarray, milliseconds: int) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).copy()
    count = min(len(samples), max(0, int(SAMPLE_RATE * milliseconds / 1000)))
    if count:
        samples[-count:] *= np.linspace(1.0, 0.0, count, dtype=np.float32)
    return samples


def _apply_poetry_master(audio: np.ndarray, delivery: dict) -> np.ndarray:
    dry = np.asarray(audio, dtype=np.float32).reshape(-1)
    wet_ratio = float(delivery["reverb_wet"])
    mixed = dry.copy()
    if wet_ratio > 0.0 and len(dry):
        wet = np.zeros_like(dry)
        pre_delay = int(SAMPLE_RATE * int(delivery["reverb_pre_delay_ms"]) / 1000)
        decay = float(delivery["reverb_decay_seconds"])
        for tap in range(1, 7):
            seconds = pre_delay / SAMPLE_RATE + decay * tap / 7.0
            delay = int(seconds * SAMPLE_RATE)
            if delay >= len(dry):
                break
            gain = math.exp(-3.0 * seconds / max(decay, 0.1))
            wet[delay:] += dry[:-delay] * gain
        wet_peak = float(np.max(np.abs(wet))) if len(wet) else 0.0
        if wet_peak > 1.0:
            wet /= wet_peak
        mixed = dry * (1.0 - wet_ratio) + wet * wet_ratio
    rms = float(np.sqrt(np.mean(np.square(mixed)) + 1e-12))
    target = 10.0 ** (float(delivery["target_rms_db"]) / 20.0)
    if rms > 0.0:
        mixed *= target / rms
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    peak_limit = 10.0 ** (float(delivery["peak_db"]) / 20.0)
    if peak > peak_limit:
        mixed *= peak_limit / peak
    return np.asarray(mixed, dtype=np.float32)


def _render_poetry(eng: Engine, req: SpeechRequest, settings: dict, delivery: dict) -> np.ndarray:
    lines = _poetry_lines(req.input)
    if not lines:
        raise HTTPException(400, "poetry input has no non-empty line")
    base_seed = int(req.seed or 0) & 0xFFFFFFFF
    output: list[np.ndarray] = []
    for line_index, (source_text, stanza_break) in enumerate(lines):
        synthesis_text = source_text
        if synthesis_text[-1:] not in ".,!?;:…":
            synthesis_text += "." if stanza_break or line_index == len(lines) - 1 else ","
        candidates: list[np.ndarray] = []
        for variant in range(int(delivery["candidates_per_line"])):
            seed = (base_seed + line_index * 1009 + variant * 9176) & 0xFFFFFFFF
            np.random.seed(seed)
            chunks = [
                np.asarray(chunk, dtype=np.float32).reshape(-1)
                for chunk in eng.tts.infer_stream(
                    synthesis_text,
                    voice=req.voice or None,
                    apply_watermark=eng.watermark,
                    **settings,
                )
                if chunk is not None and len(chunk)
            ]
            candidate = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            candidates.append(candidate)
        winner = max(candidates, key=_candidate_score)
        if _candidate_score(winner) <= -1e8:
            raise HTTPException(422, f"no valid audio candidate for poetry line {line_index + 1}")
        output.append(_fade_tail(winner, int(delivery["tail_fade_ms"])))
        if line_index < len(lines) - 1:
            if stanza_break:
                pause_ms = int(delivery["stanza_pause_ms"])
            elif source_text[-1:] in ".!?":
                pause_ms = int(delivery["sentence_pause_ms"])
            else:
                pause_ms = int(delivery["line_pause_ms"])
            if pause_ms:
                output.append(np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32))
    return _apply_poetry_master(np.concatenate(output), delivery)


def _speech_chunks(eng: Engine, req: SpeechRequest, rid: str, slot: _Slot) -> Iterator[np.ndarray]:
    """float32 chunks at ``req.sample_rate``; logs TTFA / RTF; holds one stream slot."""
    t0 = time.perf_counter()
    first = None
    emitted = 0
    rs = _Resampler(req.sample_rate)
    settings = _synthesis_settings(eng, req)
    delivery = _delivery_settings(eng, req)
    gain = 10.0 ** (float(settings.pop("gain_db", 0.0)) / 20.0)
    try:
        def generated_chunks() -> Iterator[np.ndarray]:
            if delivery["mode"] == "poetry_recitation":
                with eng._seed_lock:
                    yield _render_poetry(eng, req, settings, delivery)
                return
            if req.seed is None:
                yield from eng.tts.infer_stream(
                    req.input, voice=req.voice or None,
                    apply_watermark=eng.watermark, **settings,
                )
                return
            # VieNeu's ONNX sampler uses NumPy's process-global RNG. Seeded
            # requests are serialized so concurrent streams cannot corrupt
            # deterministic retries.
            with eng._seed_lock:
                np.random.seed(req.seed)
                yield from eng.tts.infer_stream(
                    req.input, voice=req.voice or None,
                    apply_watermark=eng.watermark, **settings,
                )

        for chunk in generated_chunks():
            if chunk is None or len(chunk) == 0:
                continue
            if gain != 1.0:
                chunk = np.clip(
                    np.asarray(chunk, dtype=np.float32) * gain,
                    -1.0,
                    1.0,
                )
            if first is None:
                first = time.perf_counter() - t0
            emitted += len(chunk)
            out = rs(chunk)
            if len(out):
                yield out
        tail = rs(np.zeros(0, np.float32), last=True)
        if len(tail):
            yield tail
    finally:
        slot.release()
        total = time.perf_counter() - t0
        audio_s = emitted / SAMPLE_RATE
        log.info("%s done: ttfa=%s total=%.2fs audio=%.2fs rtf=%s active=%d", rid,
                 f"{first*1000:.0f}ms" if first is not None else "-", total, audio_s,
                 f"{total/audio_s:.2f}" if audio_s else "-", eng.active)


def _audio_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    if fmt == "wav":
        yield _wav_header(rate)
    for c in chunks:
        yield _pcm16(c)


def _sse_body(chunks: Iterator[np.ndarray], fmt: str, rate: int) -> Iterator[bytes]:
    def ev(obj: dict) -> bytes:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")
    n = 0
    if fmt == "wav":
        yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_wav_header(rate)).decode()})
    for c in chunks:
        n += len(c)
        yield ev({"type": "speech.audio.delta", "audio": base64.b64encode(_pcm16(c)).decode()})
    yield ev({"type": "speech.audio.done",
              "usage": {"output_samples": n, "sample_rate": rate, "seconds": round(n / rate, 3)}})


@app.post("/v1/audio/speech", dependencies=[Depends(_auth)])
def speech(req: SpeechRequest):
    eng = engine()
    fmt = req.response_format.lower()
    if fmt not in ("pcm", "wav"):
        raise HTTPException(400, f"response_format '{fmt}' not supported; use 'pcm' or 'wav'")
    if req.stream_format not in ("audio", "sse"):
        raise HTTPException(400, "stream_format must be 'audio' or 'sse'")
    if req.sample_rate not in RATES:
        raise HTTPException(400, f"sample_rate must be one of {RATES}")
    if req.voice and eng.tts.resolve_voice_name(req.voice) is None:
        raise HTTPException(400, f"unknown voice '{req.voice}'; see GET /v1/voices")
    if eng.stream_error() is not None:
        # Say so now: once the 200 headers are out, a failure only truncates the body.
        raise HTTPException(503, "speech engine is down; restart the server (see /health)")
    rid = f"spk-{uuid.uuid4().hex[:8]}"
    eng.acquire()   # 429 if the server is full; released when the stream ends
    slot = _Slot(eng)
    chunks = _speech_chunks(eng, req, rid, slot)
    # A generator that never starts never runs its finally: if the client is gone
    # before the body is read, the slot is freed when the body is dropped.
    weakref.finalize(chunks, slot.release)
    headers = {"X-Request-Id": rid, "X-Sample-Rate": str(req.sample_rate), "Cache-Control": "no-store"}
    ignored = [k for k in ("speed", "instructions") if getattr(req, k) is not None]
    if ignored:
        headers["X-VieNeu-Ignored"] = ",".join(ignored)
    if req.stream_format == "sse":
        return StreamingResponse(_sse_body(chunks, fmt, req.sample_rate),
                                 media_type="text/event-stream", headers=headers)
    media = "audio/wav" if fmt == "wav" else "audio/pcm"
    return StreamingResponse(_audio_body(chunks, fmt, req.sample_rate), media_type=media, headers=headers)


# ── discovery ────────────────────────────────────────────────────────────────

@app.get("/v1/models", dependencies=[Depends(_auth)])
def models():
    eng = engine()
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "vieneu",
        "sample_rate": SAMPLE_RATE, "backend": eng.backend,
        "max_streams": eng.max_streams, "response_formats": ["pcm", "wav"],
        "stream_formats": ["audio", "sse"], "sample_rates": list(RATES),
    }]}


@app.get("/v1/voices", dependencies=[Depends(_auth)])
def voices():
    return {"object": "list", "data": engine().voices()}


# A voice name is a label: letters/digits (any script), spaces, dots, dashes,
# underscores; 1-64 chars, no control characters. It ends up in logs and in the
# voices file, so anything else is rejected at the boundary.
_VOICE_NAME = re.compile(r"[^\W_][\w .\-]{0,63}")
_CLIP_EXTS = {".wav": ".wav", ".mp3": ".mp3", ".flac": ".flac", ".ogg": ".ogg", ".m4a": ".m4a"}
_MAX_CLIP_BYTES = 20 * 1024 * 1024


def _normalize_reference_audio(
    source_path: str,
    destination_path: str,
    *,
    start_seconds: float,
    end_seconds: Optional[float],
) -> dict:
    # Lazy imports keep API contract tests independent of the production
    # runtime's native audio stack.
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    if sample_rate <= 0 or audio.size == 0:
        raise ValueError("reference audio is empty or cannot be decoded")
    total_duration = len(audio) / float(sample_rate)
    start = max(0.0, float(start_seconds or 0.0))
    end = total_duration if end_seconds is None else min(float(end_seconds), total_duration)
    if end <= start:
        raise ValueError("reference clip end must be greater than start")
    start_frame = min(len(audio), int(round(start * sample_rate)))
    end_frame = min(len(audio), int(round(end * sample_rate)))
    clip = np.mean(audio[start_frame:end_frame], axis=1, dtype=np.float32)
    duration = len(clip) / float(sample_rate)
    if duration < 3.0 or duration > 8.0:
        raise ValueError("reference clip must be between 3 and 8 seconds")
    if not np.isfinite(clip).all():
        raise ValueError("reference clip contains invalid samples")
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        clip = resample_poly(
            clip,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    if float(np.max(np.abs(clip), initial=0.0)) >= 1.0:
        raise ValueError("reference clip is clipping")
    sf.write(destination_path, clip, SAMPLE_RATE, subtype="PCM_16")
    return {
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "duration_seconds": round(len(clip) / float(SAMPLE_RATE), 6),
        "clip_start_seconds": round(start, 6),
        "clip_end_seconds": round(end, 6),
    }


@app.post("/v1/voices", dependencies=[Depends(_auth)])
def add_voice(name: str = Form(...), file: UploadFile = File(...), denoise: bool = Form(False),
              description: str = Form(""), temperature: float = Form(0.8),
              top_k: int = Form(25), top_p: float = Form(0.95),
              repetition_penalty: float = Form(1.2), max_chars: int = Form(256),
              gain_db: float = Form(0.0),
              delivery_mode: str = Form("standard"),
              candidates_per_line: int = Form(1), line_pause_ms: int = Form(0),
              sentence_pause_ms: int = Form(0), stanza_pause_ms: int = Form(0),
              tail_fade_ms: int = Form(12), target_rms_db: float = Form(-18.0),
              peak_db: float = Form(-1.0), reverb_wet: float = Form(0.0),
              reverb_pre_delay_ms: int = Form(25),
              reverb_decay_seconds: float = Form(1.1),
              start_seconds: float = Form(0.0), end_seconds: Optional[float] = Form(None),
              transcript: str = Form(""), source_label: str = Form(""),
              rights_note: str = Form("")):
    """Enroll and persist a 3-8 s reference clip as ``voice=name``.

    Plain ``def``: FastAPI runs it in a thread, so the blocking upload read,
    temp file and enrolment do not stall the event loop.
    """
    import tempfile
    eng = engine()
    if not _VOICE_NAME.fullmatch(name):
        raise HTTPException(400, "voice name: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if not _VOICE_NAME.fullmatch(description or "x"):
        raise HTTPException(400, "description: 1-64 letters, digits, spaces, '.', '-' or '_'")
    if name in eng.builtin_voices:
        # add_voice would replace it for every client, including the default voice.
        raise HTTPException(409, f"'{name}' is a built-in voice; choose another name")
    # The suffix only picks the decoder; it is mapped to a constant, never taken from the client.
    ext = _CLIP_EXTS.get(os.path.splitext(file.filename or "")[1].lower(), ".wav")
    data = file.file.read(_MAX_CLIP_BYTES + 1)
    if len(data) > _MAX_CLIP_BYTES:
        raise HTTPException(413, "reference clip larger than 20 MB")
    if len(transcript) > 4000 or len(source_label) > 300 or len(rights_note) > 1000:
        raise HTTPException(400, "voice metadata exceeds its length limit")
    fd, path = tempfile.mkstemp(suffix=ext)
    normalized_path = f"{path}.normalized.wav"
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        reference_metadata = _normalize_reference_audio(
            path,
            normalized_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        user_voices.save_user_voice(
            eng.tts,
            name,
            normalized_path,
            denoise=denoise,
            description=description,
            synthesis={
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_chars": max_chars,
                "gain_db": gain_db,
            },
            delivery={
                "mode": delivery_mode,
                "candidates_per_line": candidates_per_line,
                "line_pause_ms": line_pause_ms,
                "sentence_pause_ms": sentence_pause_ms,
                "stanza_pause_ms": stanza_pause_ms,
                "tail_fade_ms": tail_fade_ms,
                "target_rms_db": target_rms_db,
                "peak_db": peak_db,
                "reverb_wet": reverb_wet,
                "reverb_pre_delay_ms": reverb_pre_delay_ms,
                "reverb_decay_seconds": reverb_decay_seconds,
            },
            transcript=transcript,
            source_label=source_label,
            rights_note=rights_note,
            reference_metadata=reference_metadata,
        )
    except HTTPException:
        raise
    except Exception as e:   # noqa: BLE001
        raise HTTPException(400, f"could not enroll voice: {e}")
    finally:
        os.unlink(path)
        if os.path.exists(normalized_path):
            os.unlink(normalized_path)
    voice = eng.tts._preset_voices[name]
    revision = int(voice.get("revision") or 1)
    return {
        "id": name,
        "name": name,
        "description": description,
        "source": "user",
        "revision": revision,
        "synthesis": user_voices.normalize_synthesis(voice.get("synthesis")),
        "delivery": user_voices.normalize_delivery(voice.get("delivery")),
        "profile": user_voices.profile_summary(name, revision),
    }


@app.get("/v1/voices/{name}/revisions", dependencies=[Depends(_auth)])
def voice_revisions(name: str):
    voice = engine().tts._preset_voices.get(name)
    if voice is None or not voice.get(user_voices.USER_MARK):
        raise HTTPException(404, "managed voice not found")
    return {"object": "list", "data": user_voices.list_revisions(name)}


@app.get("/v1/voices/{name}/reference", dependencies=[Depends(_auth)])
def voice_reference(name: str, revision: Optional[int] = Query(None, ge=1)):
    voice = engine().tts._preset_voices.get(name)
    if voice is None or not voice.get(user_voices.USER_MARK):
        raise HTTPException(404, "managed voice not found")
    selected_revision = int(revision or voice.get("revision") or 1)
    try:
        path = user_voices.reference_audio_path(name, selected_revision)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(404, "managed voice reference not found") from exc
    return FileResponse(
        str(path),
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/v1/voices/{name}", dependencies=[Depends(_auth)])
def delete_voice(name: str):
    try:
        user_voices.delete_user_voice(engine().tts, name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": name, "deleted": True}


@app.get("/health")
def health():
    eng = engine()
    body = {"status": "ok", "backend": eng.backend, "max_streams": eng.max_streams,
            "active": eng.active, "waiting": eng.waiting, "sample_rate": SAMPLE_RATE}
    err = eng.stream_error()
    if err is not None:
        # 503 so a container healthcheck restarts us: the scheduler does not recover.
        return JSONResponse({**body, "status": "error", "error": f"stream scheduler is down: {err!r}"},
                            status_code=503)
    return body


def main() -> None:
    # Loopback unless told otherwise: the API is open when VIENEU_API_KEY is unset.
    host = os.environ.get("HOST", "127.0.0.1")
    port = _env_int("PORT", 8000)
    if host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("VIENEU_API_KEY"):
        log.warning("⚠️ listening on %s:%d without VIENEU_API_KEY: anyone who can reach this port "
                    "can use the API and enroll voices", host, port)
    # One worker: the model (and on GPU the scheduler) is process-local.
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
