#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links.model.vision import vgg


class VGG16(vgg.VGG16Layers):

    def __init__(self):
        super(VGG16, self).__init__()
        self._children.remove('fc6')
        self._children.remove('fc7')
        self._children.remove('fc8')
        del self.fc6
        del self.fc7
        del self.fc8
        del self.functions['pool5']
        del self.functions['fc6']
        del self.functions['fc7']
        del self.functions['fc8']
        del self.functions['prob']
        self.train = True

    def __call__(self, x):
        return super(VGG16, self).__call__(
            x, ['conv5_3'], test=not self.train)['conv5_3']


class VGG16Prev(chainer.Chain):

    def __init__(self, train=False):
        super(VGG16Prev, self).__init__()
        self.trunk = [
            ('conv1_1', L.Convolution2D(3, 64, 3, 1, 1)),
            ('_relu1_1', F.ReLU()),
            ('conv1_2', L.Convolution2D(64, 64, 3, 1, 1)),
            ('_relu1_2', F.ReLU()),
            ('_pool1', F.MaxPooling2D(2, 2)),
            ('conv2_1', L.Convolution2D(64, 128, 3, 1, 1)),
            ('_relu2_1', F.ReLU()),
            ('conv2_2', L.Convolution2D(128, 128, 3, 1, 1)),
            ('_relu2_2', F.ReLU()),
            ('_pool2', F.MaxPooling2D(2, 2)),
            ('conv3_1', L.Convolution2D(128, 256, 3, 1, 1)),
            ('_relu3_1', F.ReLU()),
            ('conv3_2', L.Convolution2D(256, 256, 3, 1, 1)),
            ('_relu3_2', F.ReLU()),
            ('conv3_3', L.Convolution2D(256, 256, 3, 1, 1)),
            ('_relu3_3', F.ReLU()),
            ('_pool3', F.MaxPooling2D(2, 2)),
            ('conv4_1', L.Convolution2D(256, 512, 3, 1, 1)),
            ('_relu4_1', F.ReLU()),
            ('conv4_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('_relu4_2', F.ReLU()),
            ('conv4_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('_relu4_3', F.ReLU()),
            ('_pool4', F.MaxPooling2D(2, 2)),
            ('conv5_1', L.Convolution2D(512, 512, 3, 1, 1)),
            ('_relu5_1', F.ReLU()),
            ('conv5_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('_relu5_2', F.ReLU()),
            ('conv5_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('_relu5_3', F.ReLU()),
        ]
        for name, link in self.trunk:
            if not name.startswith('_'):
                self.add_link(name, link)

    def __call__(self, x):
        for name, f in self.trunk:
            if not name.startswith('_'):
                x = getattr(self, name)(x)
            else:
                x = f(x)
            if name == '_relu5_3':
                break
        return x


if __name__ == '__main__':
    model = VGG16()
    model.train = False
    import numpy as np
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = model(x)
    print(y.shape)
