FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY app.py .
COPY src/ ./src/
COPY templates/ ./templates/
COPY data/style/ ./data/style/

# Create runtime directories
RUN mkdir -p data/uploads data/output

# Expose port
EXPOSE 5090

# Run app
CMD ["python", "app.py"]
