##
# @file Dockerfile
# @brief Dockerfile for building the sleep analysis and data collection environment.
# Uses ROCm PyTorch base image, installs dependencies, and sets up Ollama server.

# Use the PyTorch base image
FROM rocm/pytorch:latest

WORKDIR /app

# Install system dependencies for PyQt5 and OpenCV (libGL, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama server (binary)
RUN wget https://github.com/ollama/ollama/releases/download/v0.1.34/ollama-linux-amd64 -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama &&\
    file /usr/local/bin/ollama

# Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    mne \
    numpy \
    scipy \
    opencv-python \
    scikit-learn \
    PyQt5==5.15.11 \
    ollama==0.6.0 \
    beautifulsoup4==4.14.2 \
    Markdown==3.9 \
    pyserial \
    pyqtgraph

# Copy your application code into the container
COPY . .

# Expose Ollama default port
EXPOSE 11434

# Start Ollama server and your app (example: bash, or replace with your entrypoint)
CMD ["bash"]
