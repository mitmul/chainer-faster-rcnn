#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
from chainer import Variable
from chainer import serializers
from lib.models.vgg16 import VGG16

if __name__ == '__main__':
    param_fn = 'data/VGG_ILSVRC_16_layers.caffemodel'
    proto_fn = 'data/VGG_ILSVRC_16_layers_deploy.prototxt'
    net = caffe.Net(proto_fn, param_fn, caffe.TEST)

    model = VGG16()
    for name, param in net.params.iteritems():
        if 'conv' not in name:
            continue
        layer = getattr(model, name)

        print name, param[0].data.shape, param[1].data.shape,
        print layer.W.data.shape, layer.b.data.shape

        layer.W = Variable(param[0].data)
        layer.b = Variable(param[1].data)
        setattr(model, name, layer)

    serializers.save_npz('data/VGG16.model', model)
    print('data/VGG16.model saved')
