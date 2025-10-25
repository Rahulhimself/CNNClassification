FROM python:3.12-slim AS builder


WORKDIR /app


RUN apt update -y \
    && apt install -y --no-install-recommends \
        awscli \
        build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt



FROM python:3.12-slim


WORKDIR /app


COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN apt update -y \
    && apt install -y --no-install-recommends awscli \
    && rm -rf /var/lib/apt/lists/*


COPY . /app


CMD ["python3", "app.py"]