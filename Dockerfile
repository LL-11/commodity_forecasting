FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY config ./config
COPY data/rag_evaluation.jsonl ./data/rag_evaluation.jsonl
RUN python -m pip install --upgrade pip && python -m pip install ".[all]"

RUN mkdir -p /app/data
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
