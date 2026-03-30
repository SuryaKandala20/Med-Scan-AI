FROM python:3.11-slim

# System deps for OpenCV, PyMuPDF, Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist vector DB and audit DB in a volume
VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
