ARG CUDA_VERSION=10.0-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:${CUDA_VERSION}

WORKDIR /

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends curl

COPY . /galois
WORKDIR /galois
RUN curl -SL https://github.com/iedmrc/galois-autocompleter/releases/latest/download/model.tar.xz \
    | tar -xJC . && \
    pip --no-cache-dir install --upgrade pip && \
    pip --no-cache-dir install -r requirements.txt && \
    apt purge -y git curl && \
    apt autoremove --purge -y && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

CMD [ "python", "main.py" ]