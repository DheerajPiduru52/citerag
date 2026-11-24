FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    INDEX_PATH=/data/index

WORKDIR /app

# CPU-only torch first (much smaller than the default CUDA build); the
# remaining requirements then see torch as already satisfied.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake the embedding and rerank models into the image so start-up needs no download.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app/ ./app/
COPY data/sample.pdf ./data/sample.pdf

RUN useradd --create-home --uid 1000 citerag \
    && mkdir -p /data/index \
    && chown -R citerag:citerag /data /opt/hf-cache
USER citerag

VOLUME /data
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
