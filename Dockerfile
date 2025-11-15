FROM python:3.10-slim

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .
COPY faiss_index.bin .
COPY metadata.json .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]