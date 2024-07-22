FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/app

RUN apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python-is-python3 -y
RUN apt-get install pip -y
RUN apt-get install git -y
RUN apt-get install git-lfs
RUN apt-get install curl -y
RUN apt-get install ffmpeg -y
RUN apt-get install wget
RUN wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin?download=true --directory-prefix weights --content-disposition

COPY requirements.txt /usr/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow[and-cuda]
RUN pip install typing-extensions==4.9.0 --upgrade
RUN pip install git+https://github.com/tencent-ailab/IP-Adapter.git
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "3"]