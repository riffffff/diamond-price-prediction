FROM python:3.10-slim

WORKDIR /app

# Install git-lfs for large model files
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY model.pkl .
COPY encoder.pkl .
COPY features.pkl .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the Flask API
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:7860"]
