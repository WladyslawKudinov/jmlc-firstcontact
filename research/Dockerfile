FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the test suite. Override to run the pipeline, e.g.
#   docker run --rm -e XAI_API_KEY=... <image> python -m ml.generate --n 300 --out data/raw/
CMD ["pytest", "-q"]
