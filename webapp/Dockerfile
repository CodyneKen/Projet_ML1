FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY api.py .
COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "api.py", "--server.port=8081", "--server.address=0.0.0.0"]
