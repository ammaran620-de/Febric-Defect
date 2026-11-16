FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY results ./results
ENV MODEL_PATH=/app/results/best.pt
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
