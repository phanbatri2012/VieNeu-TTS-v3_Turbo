"""apps/openai_speech.py with a stand-in model: request limits, voice enrolment,
stream slots, health, auth and the listening address. ``Engine.__init__`` runs
for real."""
import gc
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps import openai_speech as api


class FakeTTS:
    backend = "onnx"

    def __init__(self):
        self._preset_voices = {"Hải Đăng": {}, "Mai Anh": {}}
        self._voice_aliases = {"Minh Quân": "Hải Đăng"}
        self.stream_calls = []
        self.enrolled = []

    def resolve_voice_name(self, name):
        if name in self._preset_voices:
            return name
        target = self._voice_aliases.get(name)
        return target if target in self._preset_voices else None

    def infer_stream(self, text, **kw):
        self.stream_calls.append(kw)
        yield np.zeros(4800, dtype=np.float32)

    def add_voice(self, name, path, **kw):
        self.enrolled.append(name)
        self._preset_voices[name] = {}

    def remove_voice(self, name, save=False):
        self._preset_voices.pop(name, None)


@pytest.fixture
def eng(monkeypatch, tmp_path):
    for var in ("VIENEU_API_KEY", "VIENEU_MAX_STREAMS", "VIENEU_QUEUE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(api, "Vieneu", lambda **kw: FakeTTS())
    monkeypatch.setenv("VIENEU_HOME", str(tmp_path))
    def fake_normalize(_source, destination, **_kwargs):
        Path(destination).write_bytes(b"RIFF")
        return {
            "sample_rate": 48000,
            "channels": 1,
            "duration_seconds": 4.0,
            "clip_start_seconds": 0.0,
            "clip_end_seconds": 4.0,
        }
    monkeypatch.setattr(api, "_normalize_reference_audio", fake_normalize)
    e = api.Engine()
    monkeypatch.setattr(api, "ENGINE", e)
    return e


@pytest.fixture
def client(eng):
    with TestClient(api.app) as c:
        yield c


def _speech(client, **body):
    return client.post("/v1/audio/speech", json={"input": "Xin chào.", **body})


def test_valid_request_streams_wav(client, eng):
    r = _speech(client, voice="Mai Anh", temperature=0.5, top_k=50)
    assert r.status_code == 200
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 4800 * 2
    assert eng.tts.stream_calls[-1]["temperature"] == 0.5
    assert eng.tts.stream_calls[-1]["top_k"] == 50
    gc.collect()                               # the drop-time release must be a no-op now
    assert eng.active == 0                     # the stream slot was given back, once


def test_unread_response_still_frees_its_slot(eng):
    # The client left while its request was queued: the body is never iterated,
    # so the stream's own finally never runs. On CPU there is a single slot, and
    # losing it meant 429 for every later request.
    resp = api.speech(api.SpeechRequest(input="Xin chào."))
    assert eng.active == 1
    del resp
    gc.collect()
    assert eng.active == 0
    assert eng._gate.acquire(blocking=False)
    eng._gate.release()


@pytest.mark.parametrize("field, value", [
    ("top_k", 10 ** 30),          # overflowed the GPU scheduler's int64 slot and killed it
    ("top_k", 0),
    ("temperature", 5.0),
    ("top_p", 0.0),
    ("repetition_penalty", 0.0),
    ("max_chars", 0),             # ZeroDivisionError in the chunker, after the 200 was sent
])
def test_out_of_range_sampling_is_rejected_up_front(client, eng, field, value):
    r = _speech(client, **{field: value})
    assert r.status_code == 400
    assert r.json()["error"]["message"].startswith(f"{field}:")   # OpenAI's error shape
    assert eng.tts.stream_calls == [{"apply_watermark": False}]   # only Engine's warm-up ran


def test_nan_temperature_is_rejected(client, eng):
    # json.loads accepts NaN; it used to reach torch.multinomial. Echoing it back
    # in the error body is what turned FastAPI's default 422 into a 500.
    r = client.post("/v1/audio/speech", content=b'{"input": "x", "temperature": NaN}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and "temperature" in r.json()["error"]["message"]


@pytest.mark.parametrize("name", ["Hải Đăng", "Minh Quân"])   # the default voice, an alias of it
def test_builtin_voice_cannot_be_replaced(client, eng, name):
    r = client.post("/v1/voices", data={"name": name}, files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 409
    assert eng.tts.enrolled == []


def test_new_voice_is_enrolled(client, eng):
    r = client.post("/v1/voices", data={"name": "Giọng của tôi"},
                    files={"file": ("me.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert eng.tts.enrolled == ["Giọng của tôi"]
    assert r.json()["source"] == "user"
    assert r.json()["revision"] == 1

    listed = client.get("/v1/voices").json()["data"]
    saved = next(item for item in listed if item["id"] == "Giọng của tôi")
    assert saved["source"] == "user"
    assert saved["synthesis"]["top_k"] == 25
    assert saved["profile"]["reference"]["available"] is True

    revisions = client.get("/v1/voices/Gi%E1%BB%8Dng%20c%E1%BB%A7a%20t%C3%B4i/revisions")
    assert revisions.status_code == 200
    assert [item["revision"] for item in revisions.json()["data"]] == [1]
    reference = client.get("/v1/voices/Gi%E1%BB%8Dng%20c%E1%BB%A7a%20t%C3%B4i/reference?revision=1")
    assert reference.status_code == 200 and reference.content == b"RIFF"


def test_voice_defaults_are_owned_by_vieneu(client, eng):
    eng.tts._preset_voices["BH_1"] = {
        api.user_voices.USER_MARK: True,
        "revision": 3,
        "synthesis": {"temperature": 0.61, "top_k": 17},
    }
    r = _speech(client, voice="BH_1")
    assert r.status_code == 200
    call = eng.tts.stream_calls[-1]
    assert call["temperature"] == 0.61
    assert call["top_k"] == 17
    assert call["top_p"] == 0.95


def test_poetry_delivery_is_owned_by_selected_voice(eng):
    eng.tts._preset_voices["BH_1"] = {
        api.user_voices.USER_MARK: True,
        "revision": 4,
        "delivery": {
            "mode": "poetry_recitation",
            "candidates_per_line": 2,
            "line_pause_ms": 900,
            "sentence_pause_ms": 1200,
            "stanza_pause_ms": 2200,
            "target_rms_db": -15,
            "reverb_wet": 0.08,
        },
    }
    delivery = api._delivery_settings(eng, api.SpeechRequest(input="Dòng một", voice="BH_1"))
    assert delivery["mode"] == "poetry_recitation"
    assert delivery["candidates_per_line"] == 2
    assert delivery["stanza_pause_ms"] == 2200
    assert api._poetry_lines("Dòng một\nDòng hai\n\nDòng ba") == [
        ("Dòng một", False),
        ("Dòng hai", True),
        ("Dòng ba", False),
    ]


def test_delete_only_removes_user_voice(client, eng):
    eng.tts._preset_voices["BH_1"] = {
        api.user_voices.USER_MARK: True,
        "revision": 1,
    }
    assert client.delete("/v1/voices/BH_1").status_code == 200
    assert "BH_1" not in eng.tts._preset_voices
    assert client.delete("/v1/voices/Mai%20Anh").status_code == 400


def test_dead_scheduler_fails_health_and_new_requests(client, eng):
    assert client.get("/health").status_code == 200
    eng.sched = SimpleNamespace(error=RuntimeError("worker died"))
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "error"
    assert _speech(client).status_code == 503
    assert eng.active == 0 and eng.waiting == 0


def test_api_key_is_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("VIENEU_API_KEY", "s3cret")
    assert client.get("/v1/voices").status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_listens_on_loopback_by_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.delenv("HOST", raising=False)
    api.main()
    assert seen["host"] == "127.0.0.1"


def test_open_public_bind_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(api.uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("VIENEU_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING, logger="vieneu.api"):
        api.main()
    assert "without VIENEU_API_KEY" in caplog.text
