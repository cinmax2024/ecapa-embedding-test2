FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN pip install --no-cache-dir runpod "numpy<2"
RUN pip install --no-cache-dir hyperpyyaml
RUN pip install --no-cache-dir --no-deps speechbrain==0.5.16
RUN pip install --no-cache-dir "huggingface-hub<1"
COPY handler.py .
CMD ["python", "-u", "handler.py"]
