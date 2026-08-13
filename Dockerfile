# syntax=docker/dockerfile:1

# Use Python 3.11 as base (within the supported range 3.10-3.13)
FROM python:3.11-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies (git needed for hip-cargo dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install uv
# Pinned rather than :latest so image builds are reproducible
COPY --from=ghcr.io/astral-sh/uv:0.9.8 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files (LICENSE is required by pyproject's license-files glob)
COPY pyproject.toml README.md LICENSE ./

# Copy source code
COPY src ./src

# Install the package with full dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system ".[full]"

# Create non-root user for running the application
RUN useradd -m -u 1000 spimple && \
    chown -R spimple:spimple /app

USER spimple

# Set the entrypoint to the spimple CLI
ENTRYPOINT ["spimple"]

# Default command shows help
CMD ["--help"]
