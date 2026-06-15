FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# OS deps for pyannote audio loading + huggingface model cache
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pre-cache python deps in their own layer for faster rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir hyperpyyaml

# ── Pre-warm: bake the models into the image ──────────────────
ARG HF_TOKEN
ENV HF_HOME=/root/.cache/huggingface
RUN test -n "${HF_TOKEN}" && \
    HF_TOKEN="${HF_TOKEN}" python -c "\
import os; \
from huggingface_hub import snapshot_download; \
snapshot_download('pyannote/speaker-diarization-3.1', token=os.environ['HF_TOKEN']); \
snapshot_download('pyannote/segmentation-3.0', token=os.environ['HF_TOKEN']); \
snapshot_download('speechbrain/spkrec-ecapa-voxceleb', token=os.environ['HF_TOKEN']); \
print('pyannote + ECAPA models baked into image')" \
    || echo "WARNING: no HF_TOKEN at build time"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
