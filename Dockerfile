# ----------------------------------------------------------------------
# STAGE 1: builder (Used for compiling dependencies and installing tools)
# ----------------------------------------------------------------------
# Use a modern, slim base image for security and smaller size
FROM python:3.12-slim-bookworm AS builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies:
# 1. awscli (Your required tool for AWS deployments)
# 2. build-essential (Needed for compiling C extensions of many Python packages)
RUN apt update -y \
    && apt install -y --no-install-recommends \
        awscli \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first.
# This maximizes caching: if only code changes, this layer is skipped.
COPY requirements.txt .

# Create and activate a virtual environment (best practice for production)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies.
# NOTE: This assumes 'requirements.txt' DOES NOT contain a reference to the
# local directory (like '-e .') which caused your previous error.
RUN pip install --no-cache-dir -r requirements.txt


# ----------------------------------------------------------------------
# STAGE 2: final (The minimal image used for production deployment)
# ----------------------------------------------------------------------
# Use a fresh, minimal image to reduce attack surface
FROM python:3.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy awscli binaries/files needed for runtime.
# This ensures awscli is available in the final slim image.
# We re-run the install to ensure the final image has the binary available.
RUN apt update -y \
    && apt install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*

# Copy the actual application code
COPY . /app

# Expose the port your application listens on (e.g., for a web service)
# EXPOSE 8080

# Define the default command to run the application when the container starts
CMD ["python3", "app.py"]