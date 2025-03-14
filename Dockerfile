# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /ap

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data/vector_stores /app/data/tmp

# Set permissions
RUN chmod -R 755 /app

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for the application
# Users should override these with their own API keys
ENV GOOGLE_API_KEY="" \
    SERPAPI_API_KEY=""

# Create a healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ -z "$GOOGLE_API_KEY" ]; then\n\
  echo "WARNING: GOOGLE_API_KEY is not set. The application requires this to function properly."\n\
fi\n\
\n\
# Add a test document if data directory is empty\n\
if [ -z "$(ls -A /app/data)" ]; then\n\
  echo "Creating sample document for testing..."\n\
  mkdir -p /app/data\n\
  echo "This is a sample document for testing the RAG system." > /app/data/sample.txt\n\
fi\n\
\n\
# Start the application\n\
exec streamlit run RAG_app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Add metadata
LABEL maintainer="Your Name <your.email@example.com>" \
      description="Advanced RAG Chatbot with Google Gemini, vectorstore, and web search capabilities" \
      version="1.0" 