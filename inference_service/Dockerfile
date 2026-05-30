# SPDX-License-Identifier: AGPL-3.0-only
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch (small, no CUDA) — the model is tiny and runs fine on CPU.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080 MODEL_NAME=erosolar-v0.01
# Single worker: the model is loaded once at import into process memory.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1
