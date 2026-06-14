import array
import audioop
import base64
import math
import os
import statistics
import tempfile
import wave

import runpod

SAMPLE_RATE = 16000

# ── Embedding mode imports (lazy, only when mode=speaker_embedding) ─
_embedding_classifier = None

def _get_embedding_classifier():
    global _embedding_classifier
    if _embedding_classifier is None:
        import functools, torch
        if not hasattr(torch.amp, 'custom_fwd'):
            def _fake_custom_fwd(fwd=None, device_type=None, cast_inputs=None):
                if fwd is None:
                    def deco(func):
                        @functools.wraps(func)
                        def w(*a, **k): return func(*a, **k)
                        return w
                    return deco
                else:
                    @functools.wraps(fwd)
                    def w(*a, **k): return fwd(*a, **k)
                    return w
            torch.amp.custom_fwd = _fake_custom_fwd
            torch.amp.custom_bwd = _fake_custom_fwd

        from speechbrain.inference.speaker import EncoderClassifier
        import torchaudio
        print("[embedding] Loading ECAPA-TDNN...", flush=True)
        _embedding_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa_model",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        print("[embedding] ECAPA loaded", flush=True)
    return _embedding_classifier


def _handle_embedding(data):
    """Handle mode=speaker_embedding: compute ECAPA embeddings and similarities."""
    import numpy as np, time, base64, tempfile, os, torch, torchaudio

    classifier = _get_embedding_classifier()
    device = next(classifier.mods.parameters()).device
    t_start = time.time()

    audio_b64 = data["audio_b64"]
    references = data.get("references") or []
    candidates = data.get("candidates") or []

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.wav")
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        print(f"[embedding] Audio: {list(waveform.shape)} @ {sr}Hz, device={device}",
              flush=True)

        def extract_clip(ws, sr, start_ms, end_ms):
            ss = int(start_ms * sr / 1000)
            es = int(end_ms * sr / 1000)
            if es > ws.shape[1]: es = ws.shape[1]
            if ss >= es or (es - ss) < 8000:  # min 0.5s at 16kHz
                return None
            clip = ws[:, ss:es]
            if sr != 16000:
                clip = torchaudio.functional.resample(clip, sr, 16000)
            return clip

        # Smoke-test with a 4s clip
        test_clip = extract_clip(waveform, sr, 1000, 5000)
        if test_clip is not None:
            test_clip = test_clip.to(device)
            with torch.no_grad():
                t0 = time.time()
                test_emb = classifier.encode_batch(test_clip.unsqueeze(0))
                enc_ms = (time.time() - t0) * 1000
            print(f"[embedding] Smoke: input={list(test_clip.shape)} "
                  f"output={list(test_emb.shape)} time={enc_ms:.0f}ms device={device}",
                  flush=True)

        # Reference embeddings
        ref_embeddings = {}
        for ref in references:
            char = ref["character"]
            segments = ref.get("segments") or []
            embs = []
            for seg in segments:
                clip = extract_clip(waveform, sr, seg["start_ms"], seg["end_ms"])
                if clip is None: continue
                clip = clip.to(device)
                with torch.no_grad():
                    emb = classifier.encode_batch(clip.unsqueeze(0))
                embs.append(emb.squeeze().cpu().numpy())
            if embs:
                ref_embeddings[char] = {
                    "mean": np.mean(embs, axis=0),
                    "count": len(embs),
                }

        # Score candidates
        def cos_sim(a, b):
            a = a.flatten(); b = b.flatten()
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        results = []
        for cand in candidates:
            clip = extract_clip(waveform, sr, cand["start_ms"], cand["end_ms"])
            if clip is None:
                results.append({"index": cand["index"], "error": "clip_too_short"})
                continue
            clip = clip.to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(clip.unsqueeze(0))
            emb = emb.squeeze().cpu().numpy()

            scores = {}
            for char, ref in ref_embeddings.items():
                scores[char] = cos_sim(emb, ref["mean"])

            sorted_chars = sorted(scores.items(), key=lambda x: -x[1])
            best = sorted_chars[0]
            second = sorted_chars[1] if len(sorted_chars) > 1 else ("", 0.0)

            results.append({
                "index": cand["index"],
                "similarities": {ch: round(s, 6) for ch, s in scores.items()},
                "best_match": best[0],
                "best_score": round(best[1], 6),
                "second_match": second[0],
                "second_score": round(second[1], 6),
                "margin": round(best[1] - second[1], 6),
            })

    elapsed = time.time() - t_start
    print(f"[embedding] Done: {len(candidates)} candidates in {elapsed:.1f}s", flush=True)
    return {
        "ok": True, "mode": "speaker_embedding",
        "device": str(device), "processing_time_s": round(elapsed, 3),
        "reference_embeddings": {ch: {"count": rd["count"]} for ch, rd in ref_embeddings.items()},
        "candidates": results,
    }


def handler(event):
    data = event.get("input") or {}

    # ── Speaker embedding mode ──────────────────────────────────────
    if data.get("mode") == "speaker_embedding":
        try:
            return _handle_embedding(data)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "retryable": False}

    # ── Standard pyannote diarization below ─────────────────────────
    try:
        audio_b64 = data["audio_b64"]
        segments = data.get("segments") or []
        settings = data.get("settings") or {}
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "audio.wav")
            with open(audio_path, "wb") as fh:
                fh.write(base64.b64decode(audio_b64))
            turns = _run_pyannote(audio_path, settings)
            mixed_by_index = _mixed_speaker_marks(segments, turns, settings)
            profiled = []
            for seg in segments:
                profile = _profile_segment(audio_path, seg, turns, settings)
                profile.update(mixed_by_index.get(int(seg.get("index") or 0), {}))
                profiled.append({**seg, **profile})
            profiled = _stabilize_speaker_profiles(profiled, settings)
            return {
                "ok": True,
                "provider": "runpod_pyannote",
                "segments": profiled,
                "turns": turns,
                "warnings": [],
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "retryable": True}


def _run_pyannote(audio_path, settings):
    """Run pyannote diarization. Loads from local cache first (no token
    required when the model was baked into the image at build time).
    Falls back to authenticated download if cache is empty. Logs every
    branch so worker logs explain why we end up with empty turns."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model = settings.get("pyannote_model") or os.environ.get(
        "PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1"
    )
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        print(f"[pyannote] FATAL: cannot import pyannote.audio: {exc}", flush=True)
        return []

    pipeline = None

    # Attempt 1: load with token (newer arg name)
    if token:
        try:
            pipeline = Pipeline.from_pretrained(model, token=token)
            print(f"[pyannote] loaded model={model} with HF_TOKEN", flush=True)
        except TypeError:
            try:
                pipeline = Pipeline.from_pretrained(model, use_auth_token=token)
                print(f"[pyannote] loaded model={model} with use_auth_token", flush=True)
            except Exception as exc:
                print(f"[pyannote] auth-load failed: {exc}", flush=True)
                pipeline = None
        except Exception as exc:
            print(f"[pyannote] token-load failed: {exc}", flush=True)
            pipeline = None

    # Attempt 2: load WITHOUT token. Works when the model was baked into
    # the image at build time (Dockerfile pre-warm) — HF cache is local,
    # no auth needed.
    if pipeline is None:
        try:
            pipeline = Pipeline.from_pretrained(model)
            print(f"[pyannote] loaded model={model} from local cache (no token needed)", flush=True)
        except Exception as exc:
            print(f"[pyannote] FATAL: model load failed both with and without token: {exc}", flush=True)
            return []

    try:
        diarization = pipeline(audio_path)
    except Exception as exc:
        print(f"[pyannote] FATAL: diarization run failed: {exc}", flush=True)
        return []

    label_map = {}
    turns = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = f"AUDIO_SPK_{len(label_map):02d}"
        turns.append(
            {
                "start_ms": int(float(turn.start) * 1000),
                "end_ms": int(float(turn.end) * 1000),
                "audio_speaker_id": label_map[label],
            }
        )
    print(f"[pyannote] returning {len(turns)} turns, {len(label_map)} unique speakers", flush=True)
    return turns


def _mixed_speaker_marks(segments, turns, settings):
    if not turns:
        return {}
    min_turn_ms = int(settings.get("mixed_speaker_min_turn_ms", 450))
    min_ratio = float(settings.get("mixed_speaker_min_turn_ratio", 0.12))
    marks = {}
    for seg in segments:
        start_ms = int(seg.get("start_ms") or 0)
        end_ms = int(seg.get("end_ms") or start_ms)
        duration = max(1, end_ms - start_ms)
        totals = {}
        overlaps = []
        for turn in turns:
            turn_start = int(turn.get("start_ms") or 0)
            turn_end = int(turn.get("end_ms") or turn_start)
            audio_id = turn.get("audio_speaker_id") or ""
            overlap = max(0, min(end_ms, turn_end) - max(start_ms, turn_start))
            if not audio_id or overlap <= 0:
                continue
            totals[audio_id] = totals.get(audio_id, 0) + overlap
            overlaps.append({
                "audio_speaker_id": audio_id,
                "start_ms": max(start_ms, turn_start),
                "end_ms": min(end_ms, turn_end),
                "overlap_ms": overlap,
            })
        strong_ids = [
            audio_id
            for audio_id, overlap in totals.items()
            if overlap >= min_turn_ms or (overlap / duration) >= min_ratio
        ]
        if len(strong_ids) > 1:
            marks[int(seg.get("index") or 0)] = {
                "mixed_speaker_detected": True,
                "mixed_speaker_count": len(strong_ids),
                "mixed_speaker_turns": overlaps,
                "speaker_profile_conflict": True,
            }
    return marks


def _profile_segment(audio_path, seg, turns, settings):
    start_ms = int(seg.get("start_ms") or 0)
    end_ms = int(seg.get("end_ms") or start_ms)
    best_turn = _best_overlap_turn(start_ms, end_ms, turns, settings)
    audio_speaker_id = best_turn["audio_speaker_id"] if best_turn else ""
    source = "runpod_pyannote" if best_turn else "runpod_pitch"
    f0_values, voiced_ms = _estimate_segment_f0(audio_path, start_ms, end_ms, settings)
    gender, confidence, median_f0 = _classify_gender(f0_values, voiced_ms, settings)
    if not audio_speaker_id and gender != "unknown":
        audio_speaker_id = f"PITCH_{gender.upper()}_00"
    if gender == "unknown":
        source = "text_fallback"
    return {
        "audio_speaker_id": audio_speaker_id,
        "audio_gender": gender,
        "audio_gender_confidence": round(confidence, 3),
        "speaker_profile_source": source,
        "speaker_profile_f0_hz": round(median_f0, 1),
        "speaker_profile_voiced_ms": int(voiced_ms),
        "speaker_profile_overlap_ms": int(best_turn.get("overlap_ms", 0)) if best_turn else 0,
        "speaker_profile_overlap_ratio": round(best_turn.get("overlap_ratio", 0.0), 3) if best_turn else 0.0,
    }


def _best_overlap_turn(start_ms, end_ms, turns, settings):
    best = None
    best_overlap = 0
    second_overlap = 0
    duration = max(1, end_ms - start_ms)
    for turn in turns:
        overlap = max(0, min(end_ms, turn["end_ms"]) - max(start_ms, turn["start_ms"]))
        if overlap > best_overlap:
            second_overlap = best_overlap
            best = turn
            best_overlap = overlap
        elif overlap > second_overlap:
            second_overlap = overlap
    min_overlap_ms = int(settings.get("min_overlap_ms", 350))
    min_overlap_ratio = float(settings.get("min_overlap_ratio", 0.30))
    min_margin_ms = int(settings.get("min_top_margin_ms", 150))
    overlap_ratio = best_overlap / duration
    if best is None:
        return None
    if best_overlap < min_overlap_ms and overlap_ratio < min_overlap_ratio:
        return None
    if second_overlap and (best_overlap - second_overlap) < min_margin_ms:
        return None
    out = dict(best)
    out["overlap_ms"] = best_overlap
    out["overlap_ratio"] = overlap_ratio
    return out


def _estimate_segment_f0(audio_path, start_ms, end_ms, settings):
    try:
        with wave.open(audio_path, "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            start_frame = max(0, int(start_ms * rate / 1000))
            end_frame = max(start_frame, int(end_ms * rate / 1000))
            wf.setpos(min(start_frame, wf.getnframes()))
            raw = wf.readframes(max(0, min(end_frame, wf.getnframes()) - start_frame))
    except Exception:
        return [], 0
    if not raw:
        return [], 0
    win = int(rate * 0.050)
    hop = int(rate * 0.100)
    min_voiced = int(settings.get("min_voiced_ms", 500))
    values = []
    voiced_ms = 0
    total_windows = max(1, (len(raw) // width - win) // max(1, hop) + 1)
    step = max(1, math.ceil(total_windows / 30))
    for pos in range(0, max(1, len(raw) // width - win + 1), hop * step):
        chunk = raw[pos * width : (pos + win) * width]
        if len(chunk) < win * width or audioop.rms(chunk, width) < 180:
            continue
        f0 = _estimate_f0(chunk, rate, width)
        if f0:
            values.append(f0)
            voiced_ms += int((hop * step) * 1000 / rate)
    if voiced_ms < min_voiced:
        return values, voiced_ms
    return values, voiced_ms


def _estimate_f0(raw, rate, width):
    samples = array.array("h")
    samples.frombytes(raw)
    if width != 2 or len(samples) < 200:
        return None
    mean = sum(samples) / len(samples)
    vals = [float(s - mean) for s in samples]
    energy = sum(v * v for v in vals)
    if energy <= 1:
        return None
    min_lag = int(rate / 400)
    max_lag = int(rate / 60)
    best_lag = 0
    best_score = 0.0
    for lag in range(min_lag, min(max_lag, len(vals) // 2)):
        corr = 0.0
        lag_energy = 0.0
        for i in range(len(vals) - lag):
            corr += vals[i] * vals[i + lag]
            lag_energy += vals[i + lag] * vals[i + lag]
        score = corr / math.sqrt(max(energy * lag_energy, 1.0))
        if score > best_score:
            best_score = score
            best_lag = lag
    return rate / best_lag if best_lag and best_score >= 0.35 else None


def _classify_gender(f0_values, voiced_ms, settings):
    min_voiced = int(settings.get("min_voiced_ms", 500))
    if not f0_values or voiced_ms < min_voiced:
        return "unknown", 0.0, 0.0
    median_f0 = statistics.median(f0_values)
    if median_f0 >= 275:
        return "child", min(0.95, 0.72 + (median_f0 - 275) / 250), median_f0
    if median_f0 >= 185:
        return "female", min(0.95, 0.70 + (median_f0 - 185) / 180), median_f0
    if median_f0 <= 155:
        return "male", min(0.95, 0.70 + (155 - median_f0) / 120), median_f0
    distance = min(abs(median_f0 - 155), abs(median_f0 - 185))
    return "unknown", max(0.35, min(0.68, distance / 30)), median_f0


def _stabilize_speaker_profiles(segments, settings):
    groups = {}
    for seg in segments:
        audio_id = seg.get("audio_speaker_id") or ""
        if audio_id.startswith("AUDIO_SPK_"):
            groups.setdefault(audio_id, []).append(seg)
    if not groups:
        return segments
    min_conf = float(settings.get("min_confidence", 0.70))
    dominance = float(settings.get("dominance_ratio", 0.72))
    stable = {}
    for audio_id, rows in groups.items():
        scores = {"male": 0.0, "female": 0.0, "child": 0.0}
        best_conf = {"male": 0.0, "female": 0.0, "child": 0.0}
        f0_values = []
        voiced_total = 0
        strong_genders = set()
        for row in rows:
            gender = str(row.get("audio_gender") or "unknown").lower()
            conf = float(row.get("audio_gender_confidence") or 0.0)
            voiced_ms = int(row.get("speaker_profile_voiced_ms") or 0)
            f0 = float(row.get("speaker_profile_f0_hz") or 0.0)
            voiced_total += max(0, voiced_ms)
            if f0 > 0 and gender != "unknown":
                f0_values.append(f0)
            if gender in scores and conf >= min_conf:
                scores[gender] += max(1, voiced_ms) * conf
                best_conf[gender] = max(best_conf[gender], conf)
                strong_genders.add(gender)
        total_score = sum(scores.values())
        top_gender = max(scores, key=scores.get)
        ratio = (scores[top_gender] / total_score) if total_score > 0 else 0.0
        conflict = len(strong_genders) > 1 and ratio < dominance
        if scores[top_gender] > 0 and ratio >= dominance and not conflict:
            gender = top_gender
            confidence = min(0.98, max(best_conf[top_gender], ratio))
        else:
            gender = "unknown"
            confidence = round(ratio, 3) if total_score > 0 else 0.0
        stable[audio_id] = {
            "gender": gender,
            "confidence": confidence,
            "conflict": conflict,
            "dominance": round(ratio, 3),
            "f0": round(statistics.median(f0_values), 1) if f0_values else 0.0,
            "voiced_ms": voiced_total,
        }
    for seg in segments:
        decision = stable.get(seg.get("audio_speaker_id") or "")
        if not decision:
            seg["speaker_profile_conflict"] = False
            continue
        seg["audio_gender_segment"] = seg.get("audio_gender", "unknown")
        seg["audio_gender_confidence_segment"] = seg.get("audio_gender_confidence", 0.0)
        seg["audio_gender"] = decision["gender"]
        seg["audio_gender_confidence"] = round(decision["confidence"], 3)
        seg["speaker_profile_source"] = "runpod_pyannote"
        seg["speaker_profile_conflict"] = decision["conflict"]
        seg["speaker_profile_dominance"] = decision["dominance"]
        seg["speaker_profile_f0_hz"] = decision["f0"]
        seg["speaker_profile_voiced_ms"] = decision["voiced_ms"]
    return segments


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
