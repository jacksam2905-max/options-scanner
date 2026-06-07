# VCP Options Scanner — cloud image (Render / any Docker host)
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    OPTIONS_SOURCE=tradier \
    OHLCV_SOURCE=tradier \
    TRADIER_BASE=https://api.tradier.com/v1

# deps first (better layer caching). lxml/pandas/numpy ship manylinux wheels,
# so no compiler needed; if a build ever fails, add: build-essential libxml2-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8787
# Render injects $PORT (serve.py reads it). HOST=0.0.0.0 binds publicly.
CMD ["python3", "dashboard/serve.py", "--no-open"]
