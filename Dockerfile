# HF Space image (PLAN_ADDENDUM 11.3). Spaces run as UID 1000 and route to 7860.
# The index is not in the image: rag/storage.py downloads it from a private
# dataset on boot (HF_DATASET_REPO + HF_TOKEN secrets).
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    DATA_DIR=/home/user/data

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so the first question isn't a download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY --chown=user rag/ ./rag/
COPY --chown=user app/ ./app/
COPY --chown=user data/manifest.yaml ./data/manifest.yaml
RUN mkdir -p /home/user/data && cp data/manifest.yaml /home/user/data/manifest.yaml

EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]
