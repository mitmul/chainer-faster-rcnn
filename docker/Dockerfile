FROM caffe:cpu

RUN apt-get update
RUN apt-get update -y && \
    apt-get install -y \
    python-dev \
    python-pip && \
    rm -rf /var/lib/apt/lists/*
RUN pip install chainer
