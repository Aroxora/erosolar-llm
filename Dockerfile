FROM python:3.14-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch numpy flask flask-cors gunicorn pyyaml

# Copy model and code
COPY model.py tokenizer.py registry.py serve.py config.py infini_attention.py ./
COPY models/registry.json ./models/
COPY models/erosolar ./models/erosolar/

# Set environment
ENV PORT=8080
ENV MODEL_PATH=/app/models/erosolar

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 serve:app
