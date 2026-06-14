"""
RunPod serverless handler — speaker embedding mode (ECAPA-TDNN).
Separate from the production diarization handler.
Deploy as a NEW endpoint; do not modify the existing pyannote handler.
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
    """Decode base64 WAV to file path, return (path, sample_rate)."""
    path = os.path.join(tmpdir, "audio.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    info = torchaudio.info(path)
    return path, info.sample_rate, info.num_frames


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
    # Ensure minimum length for Conv1d (needs ~0.5s minimum after feature extraction)
    if clip.shape[1] < 8000:  # < 0.5s at 16kHz
        return None
    return clip


def cos_sim(a, b):
    """Cosine similarity between two numpy vectors."""
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def handle_embedding(data):
    """Main embedding handler entry point."""
    classifier = get_classifier()
    device = next(classifier.mods.parameters()).device
    t_start = time.time()

    audio_b64 = data["audio_b64"]
    references = data.get("references") or []
    candidates = data.get("candidates") or []

    with tempfile.TemporaryDirectory() as tmp:
        # Decode audio
        audio_path, sr, n_frames = decode_audio(audio_b64, tmp)
        waveform, _ = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Log smoke-test diagnostics
        print(f"[embedding] Audio: {n_frames} frames @ {sr}Hz, "
              f"waveform shape={list(waveform.shape)}, device={waveform.device}",
              flush=True)

        # Extract test clip for shape verification
        test_clip = extract_clip(waveform, sr, 1000, 5000)  # 1s-5s
        if test_clip is not None:
            test_clip = test_clip.to(device)
            print(f"[embedding] Test clip shape: {list(test_clip.shape)}, device={test_clip.device}",
                  flush=True)
            with torch.no_grad():
                t_enc = time.time()
                test_emb = classifier.encode_batch(test_clip.unsqueeze(0))
                enc_time = time.time() - t_enc
            print(f"[embedding] Test embedding shape: {list(test_emb.shape)}, "
                  f"encode time: {enc_time:.3f}s", flush=True)
        else:
            print("[embedding] WARNING: test clip extraction failed", flush=True)

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
                    emb = classifier.encode_batch(clip.unsqueeze(0))
                embs.append(emb.squeeze().cpu().numpy())

            if embs:
                ref_embeddings[char] = {
                    "mean": np.mean(embs, axis=0),
                    "count": len(embs),
                    "per_segment": [e.tolist() for e in embs] if len(embs) <= 3 else [],
                }

        # ── Score candidates ─────────────────────────────────────────
        results = []
        for cand in candidates:
            clip = extract_clip(waveform, sr, cand["start_ms"], cand["end_ms"])
            if clip is None:
                results.append({
                    "index": cand["index"],
                    "error": "clip_too_short",
                })
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

    total_time = time.time() - t_start
    print(f"[embedding] Complete: {len(candidates)} candidates in {total_time:.1f}s",
          flush=True)

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


def handler(event):
    """Top-level handler — dispatches by mode."""
    data = event.get("input") or {}

    if data.get("mode") == "speaker_embedding":
        try:
            return handle_embedding(data)
        except Exception as exc:
            print(f"[embedding] ERROR: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": str(exc), "retryable": False}

    # Fallback: standard pyannote diarization (if deployed on same endpoint)
    # This path should not be used if deployed as a separate endpoint.
    return {"ok": False, "error": "unknown mode; expected mode=speaker_embedding"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
