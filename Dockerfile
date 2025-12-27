FROM python:3.13-slim

WORKDIR /model-api

COPY requirements.txt .
COPY ./src ./src

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# Jupyter Port
EXPOSE 8888 

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]