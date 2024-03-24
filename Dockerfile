FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG FACEFUSION_VERSION=2.4.1
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
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/weights/realisticVisionV60B1_v20Novae.safetensors\?download\=true --directory-prefix weights --content-disposition

RUN git clone https://github.com/facefusion/facefusion.git --branch ${FACEFUSION_VERSION} --single-branch
RUN python facefusion/install.py --onnxruntime cuda-11.8 --skip-venv

COPY requirements.txt /usr/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow[and-cuda]
RUN pip install typing-extensions==4.9.0 --upgrade
ENV MODEL_PATH="weights/realisticVisionV60B1_v20Novae.safetensors"
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "3"]