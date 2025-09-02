# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install system dependencies (if needed for nltk or others)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk vader_lexicon (required for sentiment)
RUN python -c "import nltk; nltk.download('vader_lexicon')"

# Copy the entire application code
COPY . .

# Optional: Create non-root user for security (recommended)
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /app

# Expose port 8001
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/ || exit 1

# Run the FastAPI app with uvicorn
# Use reload=True for development, remove for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]