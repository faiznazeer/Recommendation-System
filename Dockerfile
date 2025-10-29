FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs

# Cloud Run provides the port dynamically through $PORT
# So, use that instead of hardcoding 8501
ENV PORT=8080
ENV PYTHONPATH=/app

# Streamlit should bind to 0.0.0.0 and use $PORT
# Disable CORS and XSRF protection for container environments (adjust if needed)
EXPOSE 8080

CMD ["bash", "-lc", "streamlit run main.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --server.headless true"]
