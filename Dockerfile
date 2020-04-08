ARG CUDA_VERSION=10.0-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:${CUDA_VERSION}

WORKDIR /

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y curl xz-utils && \
    apt-get install -y python3 python3-pip

COPY . /galois
WORKDIR /galois

RUN curl -SL http://semantics.unisinos.br/iedmrc/galois-autocompleter/releases/latest/download/model.tar.xz \
    | tar -xJC . && \
    pip3 --no-cache-dir install --upgrade pip && \
    pip3 --no-cache-dir install -r requirements.txt && \
    apt purge -y git curl && \
    apt autoremove --purge -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

CMD [ "python3", "main.py" ]