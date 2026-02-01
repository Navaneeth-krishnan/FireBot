FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install dependencies (no dev/ml extras by default)
RUN uv sync --no-dev --extra metrics --extra viz --frozen

# Copy configs and examples
COPY configs/ configs/

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "firebot"]

# --- Development stage ---
FROM base AS dev

RUN uv sync --extra dev --extra metrics --extra viz --frozen

COPY tests/ tests/

ENTRYPOINT ["uv", "run", "pytest"]
CMD ["tests/", "-v", "--tb=short"]
