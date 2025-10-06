FROM python:3.12.5-slim AS builder

WORKDIR /app

# Install system dependencies required by rdkit and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxft2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# ---

# Stage 2: Create the final, smaller production image
FROM python:3.12.5-slim

# Set the working directory
WORKDIR /app

# Install only the necessary runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxft2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built Python packages from the builder stage
COPY --from=builder /app/wheels /wheels

# Install the Python packages from the wheels
RUN pip install --no-cache /wheels/*

# Copy the rest of your application code
COPY . .
