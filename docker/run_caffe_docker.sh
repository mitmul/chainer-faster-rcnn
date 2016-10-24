#!/bin/bash

nvidia-docker run \
-v $(readlink -f ${PWD}/../data):/workspace/data \
-v ${PWD}/../:/workspace \
--rm mitmul/chainer-faster-rcnn:caffe \
python utils/create_trunk.py
