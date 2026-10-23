# TraceGuard API. Two targets:
#   demo: small image, no data, no model, no key; serves the pre-computed memos in demo/memos.json.
#   live: CPU only; runs the agent. The data and models are NOT in the image: mount data/ and models/ read-only.
# Build: docker build --target demo -t traceguard:demo .    docker build --target live -t traceguard:live .
ARG PYTHON_VERSION=3.14

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS demo
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY src/api src/api
COPY demo demo
ENV TRACEGUARD_MODE=demo
EXPOSE 8000
USER nobody
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS live
COPY requirements-live.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-live.txt
COPY src src
COPY demo demo
COPY eval/results/threshold.json eval/results/rewrites.json eval/results/
ENV TRACEGUARD_MODE=live HF_HOME=/hf_cache
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
