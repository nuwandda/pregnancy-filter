FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

WORKDIR /usr/app

ENV PIP_DEFAULT_TIMEOUT=500

USER root
ARG DEBIAN_FRONTEND=noninteractive

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
    apt-get update && \
    apt install wget ffmpeg libsndfile1 build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
RUN apt-get install -y git
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN apt-get install git-lfs
RUN apt install curl
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib"
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/weights/realisticVisionV60B1_v20Novae.safetensors\?download\=true --directory-prefix weights --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/2dfan4.onnx\?download\=true --directory-prefix facefusion/.assets/models --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/arcface_simswap.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/face_occluder.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/face_parser.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/gender_age.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/gfpgan_1.4.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/open_nsfw.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/simswap_512_unofficial.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition
RUN wget https://huggingface.co/spaces/nuwandaa/adcreative-demo-api/resolve/main/.assets/models/yoloface_8n.onnx\?download\=true --directory-prefix facefusion/.assets/models  --content-disposition

COPY requirements.txt /usr/app/requirements.txt
RUN pip install -r requirements.txt
RUN pip install tensorflow[and-cuda]
RUN pip install typing-extensions==4.9.0 --upgrade
ENV MODEL_PATH="weights/realisticVisionV60B1_v20Novae.safetensors"
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "3"]