FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Remove the "-e ." line from requirements.txt for Docker build
RUN grep -v "^-e \." requirements.txt > requirements_docker.txt || cp requirements.txt requirements_docker.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# NOW copy application code (after dependencies are installed)
COPY flask_api/ ./flask_api/
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p logs data/raw data/interim models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=flask_api/app.py
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "flask_api/app.py"]