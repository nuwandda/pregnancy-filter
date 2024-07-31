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
RUN pip install git+https://github.com/tencent-ailab/IP-Adapter.git
RUN wget "https://civitai-delivery-worker-prod.5ac0637cfd0766c97916cefa3764fbdf.r2.cloudflarestorage.com/model/26957/realisticVisionV51.qPOH.safetensors?X-Amz-Expires=86400&response-content-disposition=attachment%3B%20filename%3D%22realisticVisionV60B1_v51HyperVAE.safetensors%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=e01358d793ad6966166af8b3064953ad/20240728/us-east-1/s3/aws4_request&X-Amz-Date=20240728T111414Z&X-Amz-SignedHeaders=host&X-Amz-Signature=180728e588b9f4bcafe45f773b1a6e6d3f86f9f1063a4f1f300d0c35c8d12c4f" --directory-prefix weights --content-disposition
COPY . .

CMD ["uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]