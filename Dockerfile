# Dockerfile for Python API
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache


EXPOSE 5000 

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --silent --fail http://localhost:5000/ia_api/health || exit 1

# Start the Flask app
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
