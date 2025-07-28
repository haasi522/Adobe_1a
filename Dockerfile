# Use Python base image for amd64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Prevent internet access during runtime
ENV NO_PROXY="*"

# Set entry point
CMD ["python", "adobe_1a.py"]
