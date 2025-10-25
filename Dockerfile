"""
FROM python:3.7-slim-buster

# Example for installing awscli
FROM debian:buster

# WARNING: Only use this method if absolutely required to use EOL Buster
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
    && sed -i '/security/d' /etc/apt/sources.list \
    && apt update -y \
    && apt install -y awscli \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
"""


# ----------------------------------------------------------------------
# STAGE 1: Build Stage (Installs packages and dependencies)
# ----------------------------------------------------------------------
# Use a current, slim base image for Python
FROM python:3.12-slim AS builder

# Set working directory for the application
WORKDIR /app

# Install the system dependencies we know we need: awscli and build tools
# We use 'apt install --no-install-recommends' for a smaller footprint
RUN apt update -y \
    && apt install -y --no-install-recommends \
        awscli \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# We use a temporary VIRTUAL_ENV to ensure a clean installation
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# ----------------------------------------------------------------------
# STAGE 2: Final Stage (Production Image)
# ----------------------------------------------------------------------
# Use a fresh, minimal runtime image for the final product
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the Python virtual environment from the builder stage
# This transfers all installed Python packages without the build-stage bloat
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The awscli installed in the builder stage is not automatically available
# in the runtime. We need to install it again or copy its binaries.
# The easiest path is re-installing just 'awscli' in the runtime.
# Since it's a critical tool for your pipeline, we include it here.
RUN apt update -y \
    && apt install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . /app

# Command to run the application
CMD ["python3", "app.py"]