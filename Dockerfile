ARG CUDA_VERSION=10.0-cudnn7-runtime-ubuntu18.04
FROM nvidia/cuda:${CUDA_VERSION}

WORKDIR /

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y curl xz-utils && \
    apt-get install -y python3 python3-pip

COPY . /galois
WORKDIR /galois

RUN pip3 --no-cache-dir install --upgrade pip && \
    pip3 --no-cache-dir install -r requirements.txt && \
    apt-get purge -y curl && \
    apt-get autoremove --purge -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD [ "python3", "main.py" ]
