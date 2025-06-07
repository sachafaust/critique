# Multi-stage build for production efficiency
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY setup.py .
COPY README.md .

# Copy source code
COPY llm_critique/ llm_critique/

# Install Python dependencies and the package
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r crituser && useradd -r -g crituser crituser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/config \
    && chown -R crituser:crituser /app

# Copy configuration files
COPY docker/config/ config/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Switch to non-root user
USER crituser

# Expose metrics port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["llm-critique", "--help"] 