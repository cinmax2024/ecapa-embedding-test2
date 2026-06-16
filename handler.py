"""
RunPod serverless handler — speaker embedding + compact LOO evaluation (ECAPA-TDNN).
Separate from the production diarization handler.
Deploy as a standalone endpoint; does not require pyannote.
"""

import base64
import functools
import os
import tempfile
import time
import wave

import numpy as np
import runpod

# ── Monkey-patch for torch 2.1.2 + SpeechBrain compatibility ──────
import torch
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

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# ── Model cache ────────────────────────────────────────────────────
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        print("[embedding] Loading ECAPA-TDNN model...", flush=True)
        t0 = time.time()
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa_model",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
        print(f"[embedding] Model loaded in {time.time()-t0:.1f}s", flush=True)
    return _classifier


def decode_audio(audio_b64, tmpdir):
    """Decode base64 WAV to file path, return (path, sample_rate, num_frames)."""
    path = os.path.join(tmpdir, "audio.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    info = torchaudio.info(path)
    return path, info.sample_rate, info.num_frames


def load_audio(data, tmpdir):
    """Load audio from audio_b64 or audio_url, return (path, sr, n_frames)."""
    audio_url = data.get("audio_url")
    if audio_url:
        import urllib.request
        path = os.path.join(tmpdir, "audio.wav")
        print(f"[embedding] Downloading audio from {audio_url}...", flush=True)
        urllib.request.urlretrieve(audio_url, path)
        info = torchaudio.info(path)
        return path, info.sample_rate, info.num_frames
    else:
        return decode_audio(data["audio_b64"], tmpdir)


def extract_clip(waveform, sr, start_ms, end_ms):
    """Extract audio segment and resample to 16kHz mono for ECAPA."""
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
    if start_sample >= end_sample:
        return None
    clip = waveform[:, start_sample:end_sample]
    if sr != 16000:
        clip = torchaudio.functional.resample(clip, sr, 16000)
    if clip.shape[1] < 8000:  # < 0.5s at 16kHz
        return None
    return clip


def cos_sim(a, b):
    """Cosine similarity between two numpy vectors."""
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def handle_embedding(data):
    """speaker_embedding mode: compute reference embeddings and score candidates."""
    classifier = get_classifier()
    device = next(classifier.mods.parameters()).device
    t_start = time.time()

    references = data.get("references") or []
    candidates = data.get("candidates") or []

    with tempfile.TemporaryDirectory() as tmp:
        audio_path, sr, n_frames = load_audio(data, tmp)
        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        print(f"[embedding] Audio: {n_frames} frames @ {sr}Hz, "
              f"waveform shape={list(waveform.shape)}, device={device}",
              flush=True)

        # Smoke-test: encode a 4s clip to verify model health
        test_clip = extract_clip(waveform, sr, 1000, 5000)
        if test_clip is not None:
            test_clip = test_clip.to(device)
            with torch.no_grad():
                t_enc = time.time()
                test_emb = classifier.encode_batch(test_clip)
                enc_time = time.time() - t_enc
            print(f"[embedding] Test embedding shape: {list(test_emb.shape)}, "
                  f"encode time: {enc_time:.3f}s", flush=True)

        # ── Compute reference embeddings ─────────────────────────────
        ref_embeddings = {}
        for ref in references:
            char = ref["character"]
            segments = ref.get("segments") or []
            embs = []
            for seg in segments:
                clip = extract_clip(waveform, sr, seg["start_ms"], seg["end_ms"])
                if clip is None:
                    continue
                clip = clip.to(device)
                with torch.no_grad():
                    emb = classifier.encode_batch(clip)
                embs.append(emb.squeeze().cpu().numpy())

            if embs:
                ref_embeddings[char] = {
                    "mean": np.mean(embs, axis=0),
                    "count": len(embs),
                }

        # ── Score candidates ─────────────────────────────────────────
        results = []
        for cand in candidates:
            clip = extract_clip(waveform, sr, cand["start_ms"], cand["end_ms"])
            if clip is None:
                results.append({"index": cand["index"], "error": "clip_too_short"})
                continue

            clip = clip.to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(clip)
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

    total_time = time.time() - t_start
    print(f"[embedding] Complete: {len(candidates)} candidates in {total_time:.1f}s", flush=True)

    return {
        "ok": True,
        "mode": "speaker_embedding",
        "device": str(device),
        "processing_time_s": round(total_time, 3),
        "reference_embeddings": {
            ch: {"count": rd["count"]}
            for ch, rd in ref_embeddings.items()
        },
        "candidates": results,
    }


def handle_loo_evaluation(data):
    """loo_evaluation mode: compact LOO with pre-aligned anchor clips.
    
    Accepts: audio_b64 or audio_url, anchors: [{index, character, start_ms, end_ms}]
    Encodes all anchors once, runs LOO at multiple thresholds, returns metrics.
    """
    from collections import defaultdict

    classifier = get_classifier()
    device = next(classifier.mods.parameters()).device
    t_start = time.time()

    anchors = data.get("anchors") or []
    if not anchors:
        return {"ok": False, "error": "no anchors provided", "retryable": False}

    with tempfile.TemporaryDirectory() as tmp:
        audio_path, sr, n_frames = load_audio(data, tmp)
        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        print(f"[loo] Audio: {n_frames} frames @ {sr}Hz, {len(anchors)} anchors, device={device}",
              flush=True)

        # Step 1: Encode all anchors
        print("[loo] Encoding all anchors...", flush=True)
        t_enc = time.time()
        embeddings = {}  # index -> numpy embedding
        for a in anchors:
            clip = extract_clip(waveform, sr, a["start_ms"], a["end_ms"])
            if clip is None:
                embeddings[a["index"]] = None
                continue
            clip = clip.to(device)
            with torch.no_grad():
                emb = classifier.encode_batch(clip)
            embeddings[a["index"]] = emb.squeeze().cpu().numpy()

        valid = sum(1 for e in embeddings.values() if e is not None)
        enc_time = time.time() - t_enc
        print(f"[loo] Encoded {valid}/{len(anchors)} in {enc_time:.1f}s", flush=True)

        if valid == 0:
            return {"ok": False, "error": "no valid anchor clips extracted", "retryable": False}

        # Step 2: Group by character
        by_char = defaultdict(list)
        for a in anchors:
            by_char[a["character"]].append(a)

        # Step 3: Run LOO at multiple thresholds
        THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        MARGINS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]

        all_results = []
        print("[loo] Computing LOO...", flush=True)

        for st in THRESHOLDS:
            for mg in MARGINS:
                correct = 0
                incorrect = 0
                rejected = 0
                confusions = defaultdict(lambda: defaultdict(int))

                for a in anchors:
                    e = embeddings.get(a["index"])
                    if e is None:
                        rejected += 1
                        continue
                    tc = a["character"]

                    # Build per-character mean embeddings EXCLUDING this anchor
                    char_means = {}
                    for ch, lst in by_char.items():
                        other_embs = [embeddings[aa["index"]] for aa in lst
                                      if aa["index"] != a["index"] and embeddings.get(aa["index"]) is not None]
                        if other_embs:
                            char_means[ch] = np.mean(other_embs, axis=0)

                    if len(char_means) < 2:
                        rejected += 1
                        continue

                    scores = {ch: cos_sim(e, m) for ch, m in char_means.items()}
                    sorted_ch = sorted(scores.items(), key=lambda x: -x[1])
                    best_ch, best_score = sorted_ch[0]
                    second_score = sorted_ch[1][1] if len(sorted_ch) > 1 else 0.0

                    if best_score < st or (best_score - second_score + 1e-9) < mg:
                        rejected += 1
                    elif best_ch == tc:
                        correct += 1
                    else:
                        incorrect += 1
                        confusions[tc][best_ch] += 1

                total = correct + incorrect + rejected
                if total == 0:
                    continue
                prec = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
                cov = correct / total
                f1 = 2 * prec * cov / (prec + cov) if (prec + cov) > 0 else 0

                all_results.append({
                    "similarity_threshold": round(st, 2),
                    "margin": round(mg, 2),
                    "correct": correct,
                    "incorrect": incorrect,
                    "rejected": rejected,
                    "precision": round(prec, 4),
                    "coverage": round(cov, 4),
                    "f1": round(f1, 4),
                    "confusion_pairs": {tc: dict(pm) for tc, pm in confusions.items() if pm},
                })

    total_time = time.time() - t_start
    print(f"[loo] Done in {total_time:.1f}s", flush=True)

    # Find best results
    best_f1 = max(all_results, key=lambda r: r["f1"])
    best_prec = max([r for r in all_results if r["precision"] >= 0.95],
                    key=lambda r: r["coverage"], default=best_f1)

    return {
        "ok": True,
        "mode": "loo_evaluation",
        "device": str(device),
        "total_anchors": len(anchors),
        "valid_anchors": valid,
        "encoding_time_s": round(enc_time, 3),
        "processing_time_s": round(total_time, 3),
        "best_f1": best_f1,
        "best_precision_ge_095": best_prec,
        "all_results": all_results,
    }


def handler(event):
    """Top-level handler — dispatches by mode."""
    data = event.get("input") or {}

    if data.get("mode") == "speaker_embedding":
        try:
            return handle_embedding(data)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "retryable": False}

    if data.get("mode") == "loo_evaluation":
        try:
            return handle_loo_evaluation(data)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "retryable": False}

    return {"ok": False, "error": f"unknown mode: {data.get('mode')}. Expected speaker_embedding or loo_evaluation"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
