FROM python:3.12.5-slim

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies required by rdkit.
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container.
COPY . .
