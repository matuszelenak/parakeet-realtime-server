FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    build-essential \
    python3 \
    python3-pip \
    ffmpeg

WORKDIR /app

RUN pip install torch --break-system-packages
RUN pip install nemo_toolkit['asr'] --break-system-packages
RUN pip install cuda-python>=12.3 numpy==1.26.4 --break-system-packages

COPY requirements.txt /app/requirements.txt
RUN pip install --break-system-packages -r /app/requirements.txt

ENTRYPOINT ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9000", "--timeout-graceful-shutdown", "0", "--reload"]