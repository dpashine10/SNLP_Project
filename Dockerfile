FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies before copying the source so Docker can reuse this
# expensive layer when only application files change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    GROUP6_URL=http://group6:5000

EXPOSE 8000

CMD ["uv", "run", "python", "app.py"]
