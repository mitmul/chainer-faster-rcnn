#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.Chain):

    def __init__(self, train=False):
        super(VGG16, self).__init__()
        self.trunk = [
            ('conv1_1', L.Convolution2D(3, 64, 3, 1, 1)),
            ('relu1_1', F.ReLU()),
            ('conv1_2', L.Convolution2D(64, 64, 3, 1, 1)),
            ('relu1_2', F.ReLU()),
            ('pool1', F.MaxPooling2D(2, 2)),
            ('conv2_1', L.Convolution2D(64, 128, 3, 1, 1)),
            ('relu2_1', F.ReLU()),
            ('conv2_2', L.Convolution2D(128, 128, 3, 1, 1)),
            ('relu2_2', F.ReLU()),
            ('pool2', F.MaxPooling2D(2, 2)),
            ('conv3_1', L.Convolution2D(128, 256, 3, 1, 1)),
            ('relu3_1', F.ReLU()),
            ('conv3_2', L.Convolution2D(256, 256, 3, 1, 1)),
            ('relu3_2', F.ReLU()),
            ('conv3_3', L.Convolution2D(256, 256, 3, 1, 1)),
            ('relu3_3', F.ReLU()),
            ('pool3', F.MaxPooling2D(2, 2)),
            ('conv4_1', L.Convolution2D(256, 512, 3, 1, 1)),
            ('relu4_1', F.ReLU()),
            ('conv4_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu4_2', F.ReLU()),
            ('conv4_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu4_3', F.ReLU()),
            ('pool4', F.MaxPooling2D(2, 2)),
            ('conv5_1', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_1', F.ReLU()),
            ('conv5_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_2', F.ReLU()),
            ('conv5_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_3', F.ReLU()),
            ('rpn_conv_3x3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('rpn_relu_3x3', F.ReLU()),
        ]
        for name, link in self.trunk:
            if 'conv' in name:
                self.add_link(name, link)

    def __call__(self, x):
        for name, f in self.trunk:
            x = (getattr(self, name) if 'conv' in name else f)(x)
            if 'relu5_3' in name:
                self.feature = x
        return x
