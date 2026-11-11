FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY --chown=app:app . .

USER app
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
