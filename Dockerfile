# Use the official Python image
FROM python:3.12.5-slim

# Install a comprehensive set of dependencies for rdkit's graphical operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libxrender1 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxft2 \
    libcairo2 \
    libcairo2-dev \
    libpango1.0-0 \
    libpangocairo-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Set the command to run the application using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
