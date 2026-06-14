# Pyannote Speaker Profiler — RunPod Serverless

Speaker diarization for the AddisTV dubbing pipeline. Runs as a RunPod
serverless endpoint and is called from the dashboard server's
`speaker_profiler._run_runpod_profile()`.

## What's in here

| File | Purpose |
|---|---|
| `handler.py` | The RunPod handler — receives `{audio_b64, segments, settings}`, runs pyannote diarization + per-segment pitch profiling, returns enriched segments. Ends with `runpod.serverless.start(...)` so it boots as a serverless worker. |
| `Dockerfile` | CUDA + pyannote.audio + ffmpeg + libsndfile. CMD runs the handler. |
| `requirements.txt` | Pinned: `runpod==1.7.9`, `pyannote.audio==3.1.1`, etc. |

## How RunPod uses this

1. RunPod auto-builds the Docker image when you point an endpoint at this repo.
2. On every incoming request, RunPod starts a container, calls `handler(event)`.
3. `handler` reads the base64 audio, runs the pyannote pipeline, returns JSON.

## Required environment variables on the RunPod endpoint

Set these in the endpoint config (Manage → Environment Variables):

| Key | Value | Why |
|---|---|---|
| `HF_TOKEN` | the same `hf_...` token used elsewhere | pyannote model is gated; pipeline can't load without it |
| `PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | (optional, this is the default) |

## Local sanity check (optional)

```bash
docker build -t pyannote-profile .
docker run --rm -e HF_TOKEN=hf_xxx pyannote-profile python -c \
  "import handler; print('handler imported ok, runpod sdk linked')"
```

## Connection to the dubbing pipeline

The dashboard server's `.env` points at this endpoint:
```
RUNPOD_SPEAKER_PROFILE_ENDPOINT=https://api.runpod.ai/v2/<endpoint-id>/runsync
RUNPOD_API_KEY=rpa_...
SPEAKER_PROFILING_PROVIDER=runpod
```

The dashboard's `speaker_profiler.py` posts `{audio_b64, segments, settings}`
and parses the result via `apply_runpod_profile_output()`.
