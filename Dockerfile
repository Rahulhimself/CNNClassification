FROM python:3.7-slim-buster

# Example for installing awscli
RUN apt update -y \ 
    && apt install -y awscli \ 
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]