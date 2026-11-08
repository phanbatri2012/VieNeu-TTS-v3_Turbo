"""Voices the user saved from the Voice Cloning tab.

The SDK's ``add_voice`` registers a clone for the running session and its
``save_voices`` rewrites the *built-in* preset file inside the package, which an
upgrade would wipe. So the app keeps its own file, only with the voices the user
added, and loads it on top of the built-ins every time a v3 model is loaded:

    ~/.vieneu/user_voices_v3_turbo.json   (v3 Turbo: speaker_emb + codes)
    ~/.vieneu/user_voices_v3_nano.json    (v3 Nano:  speaker_emb + style)

``VIENEU_HOME`` overrides the folder. Saved voices show up in every preset
dropdown (story, conversation, SRT) and survive restarts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

USER_MARK = "_user"          # key set on tts._preset_voices entries that came from here
DEFAULT_DESC = "giọng đã lưu"
DEFAULT_SYNTHESIS = {
    "temperature": 0.8,
    "top_k": 25,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "max_chars": 256,
    "gain_db": 0.0,
}
DEFAULT_DELIVERY = {
    "mode": "standard",
    "candidates_per_line": 1,
    "line_pause_ms": 0,
    "sentence_pause_ms": 0,
    "stanza_pause_ms": 0,
    "tail_fade_ms": 12,
    "target_rms_db": -18.0,
    "peak_db": -1.0,
    "reverb_wet": 0.0,
    "reverb_pre_delay_ms": 25,
    "reverb_decay_seconds": 1.1,
}


def _kind(tts) -> str:
    return "nano" if "nano" in type(tts).__name__.lower() else "turbo"


def voices_home() -> Path:
    return Path(os.environ.get("VIENEU_HOME") or (Path.home() / ".vieneu"))


def user_voices_path(tts) -> Path:
    return voices_home() / f"user_voices_v3_{_kind(tts)}.json"


def voice_profile_directory(name: str) -> Path:
    storage_id = hashlib.sha256(str(name or "").encode("utf-8")).hexdigest()
    return voices_home() / "profiles" / storage_id


def revision_directory(name: str, revision: int) -> Path:
    return voice_profile_directory(name) / "revisions" / str(max(1, int(revision)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def list_revisions(name: str) -> list[dict]:
    root = voice_profile_directory(name) / "revisions"
    if not root.is_dir():
        return []
    revisions = []
    for directory in sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        try:
            revisions.append(profile_summary(name, int(directory.name)))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return revisions


def profile_summary(name: str, revision: int) -> dict:
    directory = revision_directory(name, revision)
    metadata = _read_json(directory / "metadata.json")
    checksums = _read_json(directory / "checksums.json").get("files") or {}
    reference = directory / "reference.wav"
    expected = str(checksums.get("reference.wav") or "").casefold()
    if not reference.is_file() or not expected or _sha256(reference) != expected:
        raise ValueError("managed voice reference checksum mismatch")
    return {
        "voice_id": str(metadata.get("voice_id") or name),
        "revision": int(metadata.get("revision") or revision),
        "description": str(metadata.get("description") or ""),
        "transcript": str(metadata.get("transcript") or ""),
        "source_label": str(metadata.get("source_label") or ""),
        "rights_note": str(metadata.get("rights_note") or ""),
        "created_at": str(metadata.get("created_at") or ""),
        "synthesis": normalize_synthesis(metadata.get("synthesis")),
        "delivery": normalize_delivery(metadata.get("delivery")),
        "reference": {
            "available": True,
            "sha256": expected,
            "sample_rate": int(metadata.get("sample_rate") or 0),
            "channels": int(metadata.get("channels") or 0),
            "duration_seconds": float(metadata.get("duration_seconds") or 0),
            "clip_start_seconds": float(metadata.get("clip_start_seconds") or 0),
            "clip_end_seconds": float(metadata.get("clip_end_seconds") or 0),
        },
    }


def reference_audio_path(name: str, revision: int) -> Path:
    profile_summary(name, revision)
    return revision_directory(name, revision) / "reference.wav"


def supports_saving(tts) -> bool:
    return tts is not None and hasattr(tts, "add_voice") and hasattr(tts, "_preset_voices")


def list_user_voices(tts) -> list[str]:
    if tts is None or not hasattr(tts, "_preset_voices"):
        return []
    return [n for n, v in tts._preset_voices.items() if isinstance(v, dict) and v.get(USER_MARK)]


def _entry_to_json(v: dict) -> dict:
    out: dict[str, Any] = {
        "description": v.get("description", ""),
        "gender": v.get("gender", ""),
        "revision": max(1, int(v.get("revision") or 1)),
        "synthesis": normalize_synthesis(v.get("synthesis")),
        "delivery": normalize_delivery(v.get("delivery")),
    }
    emb = v.get("speaker_emb")
    out["speaker_emb"] = [round(float(x), 6) for x in np.asarray(emb, dtype=np.float32).reshape(-1)] if emb is not None else None
    if "style" in v and isinstance(v["style"], np.ndarray):          # Nano: style tokens [50, 256]
        out["style"] = np.asarray(v["style"], dtype=np.float32).round(5).tolist()
    else:                                                            # Turbo: reference codes
        codes = v.get("codes")
        out["codes"] = None if codes is None else np.asarray(codes, dtype=np.int64).tolist()
        if isinstance(v.get("style"), str):
            out["style_name"] = v["style"]
    return out


def _entry_from_json(tts, v: dict) -> Optional[dict]:
    emb = v.get("speaker_emb")
    if emb is None:
        return None
    entry: dict[str, Any] = {
        "description": v.get("description", "") or DEFAULT_DESC,
        "gender": v.get("gender", ""),
        "speaker_emb": np.asarray(emb, dtype=np.float32),
        "podcast": True,
        "revision": max(1, int(v.get("revision") or 1)),
        "synthesis": normalize_synthesis(v.get("synthesis")),
        "delivery": normalize_delivery(v.get("delivery")),
        USER_MARK: True,
    }
    if _kind(tts) == "nano":
        if v.get("style") is None:
            return None
        style = np.asarray(v["style"], dtype=np.float32)
        entry["style"] = style
        entry["codes"] = style
    else:
        codes = v.get("codes")
        entry["codes"] = None if codes is None else np.asarray(codes, dtype=np.int64)
        entry["style"] = v.get("style_name") or getattr(tts, "default_style", "tu_nhien")
    return entry


def load_user_voices(tts) -> list[str]:
    """Add the saved voices to ``tts._preset_voices``; returns the names loaded.
    Never raises — a broken file only means no user voices this session."""
    if not supports_saving(tts):
        return []
    path = user_voices_path(tts)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    names: list[str] = []
    for name, v in (data.get("presets") or {}).items():
        entry = _entry_from_json(tts, v) if isinstance(v, dict) else None
        if entry is None:
            continue
        tts._preset_voices[name] = entry
        names.append(name)
    return names


def _write(tts) -> Path:
    path = user_voices_path(tts)
    path.parent.mkdir(parents=True, exist_ok=True)
    presets = {n: _entry_to_json(tts._preset_voices[n]) for n in list_user_voices(tts)}
    data = {"meta": {"note": f"VieNeu v3 {_kind(tts)} voices saved from the Gradio app"}, "presets": presets}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def normalize_synthesis(value: Any) -> dict:
    raw = value if isinstance(value, dict) else {}
    synthesis = {
        "temperature": float(raw.get("temperature", DEFAULT_SYNTHESIS["temperature"])),
        "top_k": int(raw.get("top_k", DEFAULT_SYNTHESIS["top_k"])),
        "top_p": float(raw.get("top_p", DEFAULT_SYNTHESIS["top_p"])),
        "repetition_penalty": float(
            raw.get("repetition_penalty", DEFAULT_SYNTHESIS["repetition_penalty"])
        ),
        "max_chars": int(raw.get("max_chars", DEFAULT_SYNTHESIS["max_chars"])),
        "gain_db": float(raw.get("gain_db", DEFAULT_SYNTHESIS["gain_db"])),
    }
    if not 0.0 <= synthesis["temperature"] <= 2.0:
        raise ValueError("temperature phải nằm trong khoảng 0-2.")
    if not 1 <= synthesis["top_k"] <= 1024:
        raise ValueError("top_k phải nằm trong khoảng 1-1024.")
    if not 0.0 < synthesis["top_p"] <= 1.0:
        raise ValueError("top_p phải lớn hơn 0 và không vượt quá 1.")
    if not 1.0 <= synthesis["repetition_penalty"] <= 2.0:
        raise ValueError("repetition_penalty phải nằm trong khoảng 1-2.")
    if not 64 <= synthesis["max_chars"] <= 512:
        raise ValueError("max_chars phải nằm trong khoảng 64-512.")
    if not -12.0 <= synthesis["gain_db"] <= 12.0:
        raise ValueError("gain_db phải nằm trong khoảng -12 đến 12 dB.")
    return synthesis


def normalize_delivery(value: Any) -> dict:
    raw = value if isinstance(value, dict) else {}
    mode = str(raw.get("mode") or DEFAULT_DELIVERY["mode"]).strip().casefold()
    if mode not in {"standard", "poetry_recitation"}:
        raise ValueError("delivery mode phải là standard hoặc poetry_recitation.")
    delivery = {
        "mode": mode,
        "candidates_per_line": int(
            raw.get("candidates_per_line", DEFAULT_DELIVERY["candidates_per_line"])
        ),
        "line_pause_ms": int(raw.get("line_pause_ms", DEFAULT_DELIVERY["line_pause_ms"])),
        "sentence_pause_ms": int(
            raw.get("sentence_pause_ms", DEFAULT_DELIVERY["sentence_pause_ms"])
        ),
        "stanza_pause_ms": int(
            raw.get("stanza_pause_ms", DEFAULT_DELIVERY["stanza_pause_ms"])
        ),
        "tail_fade_ms": int(raw.get("tail_fade_ms", DEFAULT_DELIVERY["tail_fade_ms"])),
        "target_rms_db": float(
            raw.get("target_rms_db", DEFAULT_DELIVERY["target_rms_db"])
        ),
        "peak_db": float(raw.get("peak_db", DEFAULT_DELIVERY["peak_db"])),
        "reverb_wet": float(raw.get("reverb_wet", DEFAULT_DELIVERY["reverb_wet"])),
        "reverb_pre_delay_ms": int(
            raw.get("reverb_pre_delay_ms", DEFAULT_DELIVERY["reverb_pre_delay_ms"])
        ),
        "reverb_decay_seconds": float(
            raw.get("reverb_decay_seconds", DEFAULT_DELIVERY["reverb_decay_seconds"])
        ),
    }
    if not 1 <= delivery["candidates_per_line"] <= 3:
        raise ValueError("candidates_per_line phải nằm trong khoảng 1-3.")
    for key in ("line_pause_ms", "sentence_pause_ms", "stanza_pause_ms"):
        if not 0 <= delivery[key] <= 5000:
            raise ValueError(f"{key} phải nằm trong khoảng 0-5000 ms.")
    if not 0 <= delivery["tail_fade_ms"] <= 100:
        raise ValueError("tail_fade_ms phải nằm trong khoảng 0-100 ms.")
    if not -30.0 <= delivery["target_rms_db"] <= -8.0:
        raise ValueError("target_rms_db phải nằm trong khoảng -30 đến -8 dB.")
    if not -6.0 <= delivery["peak_db"] <= -0.1:
        raise ValueError("peak_db phải nằm trong khoảng -6 đến -0.1 dB.")
    if not 0.0 <= delivery["reverb_wet"] <= 0.25:
        raise ValueError("reverb_wet phải nằm trong khoảng 0-0.25.")
    if not 0 <= delivery["reverb_pre_delay_ms"] <= 200:
        raise ValueError("reverb_pre_delay_ms phải nằm trong khoảng 0-200 ms.")
    if not 0.1 <= delivery["reverb_decay_seconds"] <= 3.0:
        raise ValueError("reverb_decay_seconds phải nằm trong khoảng 0.1-3 giây.")
    if mode == "standard":
        delivery.update(
            {
                "candidates_per_line": 1,
                "line_pause_ms": 0,
                "sentence_pause_ms": 0,
                "stanza_pause_ms": 0,
                "reverb_wet": 0.0,
            }
        )
    return delivery


def save_user_voice(
    tts,
    name: str,
    ref_audio: str,
    *,
    denoise: bool = True,
    description: str = "",
    synthesis: Optional[dict] = None,
    delivery: Optional[dict] = None,
    transcript: str = "",
    source_label: str = "",
    rights_note: str = "",
    reference_metadata: Optional[dict] = None,
) -> str:
    """Enroll ``ref_audio`` as a named preset and persist it. Returns the voice id.
    A name that belongs to a built-in voice is refused; re-saving one of the
    user's own names replaces it."""
    if not supports_saving(tts):
        raise ValueError("Chỉ VieNeu v3 (Turbo / Nano) mới lưu được giọng.")
    name = (name or "").strip()
    if not name:
        raise ValueError("Hãy đặt tên cho giọng.")
    if len(name) > 40:
        raise ValueError("Tên giọng tối đa 40 ký tự.")
    if "—" in name:
        raise ValueError("Tên giọng không được chứa dấu gạch dài (—).")
    if not ref_audio:
        raise ValueError("Hãy tải lên audio mẫu trước.")
    existing = tts._preset_voices.get(name)
    if existing is not None and not existing.get(USER_MARK):
        raise ValueError(f"'{name}' là giọng có sẵn của VieNeu, hãy chọn tên khác.")
    revision = max(1, int((existing or {}).get("revision") or 0) + 1)
    normalized_synthesis = normalize_synthesis(synthesis)
    normalized_delivery = normalize_delivery(delivery)
    reference = Path(ref_audio)
    if not reference.is_file():
        raise ValueError("Audio mẫu không tồn tại.")
    clean_description = (description or "").strip() or DEFAULT_DESC
    destination = revision_directory(name, revision)
    if destination.exists():
        raise ValueError("Revision giọng đã tồn tại.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".revision-", dir=str(destination.parent)))
    previous = copy.deepcopy(existing) if existing is not None else None
    try:
        tts.add_voice(name, str(reference), denoise=denoise, description=clean_description)
        entry = tts._preset_voices[name]
        entry[USER_MARK] = True
        entry["podcast"] = True
        entry["revision"] = revision
        entry["synthesis"] = normalized_synthesis
        entry["delivery"] = normalized_delivery

        managed_reference = temporary / "reference.wav"
        shutil.copy2(reference, managed_reference)
        (temporary / "transcript.txt").write_text(str(transcript or ""), encoding="utf-8")
        details = dict(reference_metadata or {})
        metadata = {
            "voice_id": name,
            "revision": revision,
            "description": clean_description,
            "transcript": str(transcript or ""),
            "source_label": str(source_label or ""),
            "rights_note": str(rights_note or ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sample_rate": int(details.get("sample_rate") or 0),
            "channels": int(details.get("channels") or 0),
            "duration_seconds": float(details.get("duration_seconds") or 0),
            "clip_start_seconds": float(details.get("clip_start_seconds") or 0),
            "clip_end_seconds": float(details.get("clip_end_seconds") or 0),
            "synthesis": normalized_synthesis,
            "delivery": normalized_delivery,
            "denoise": bool(denoise),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        checksums = {
            "files": {
                filename: _sha256(temporary / filename)
                for filename in ("reference.wav", "transcript.txt", "metadata.json")
            }
        }
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, destination)
        _write(tts)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)
        if previous is None:
            tts._preset_voices.pop(name, None)
        else:
            tts._preset_voices[name] = previous
        raise
    return name


def delete_user_voice(tts, name: str) -> None:
    if not supports_saving(tts) or not name:
        return
    entry = tts._preset_voices.get(name)
    if entry is None or not entry.get(USER_MARK):
        raise ValueError("Chỉ xoá được giọng do bạn lưu.")
    tts.remove_voice(name)
    _write(tts)
    shutil.rmtree(voice_profile_directory(name), ignore_errors=True)
