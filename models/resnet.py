#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

from chainer.dataset import download
from chainer.links.model.vision import resnet


class ResNet(resnet.ResNetLayers):

    URLS = {
        'resnet50': 'https://www.dropbox.com/s/f0sqe7za120msbs/'
                    'ResNet-50-model.caffemodel?dl=1',
        'resnet101': 'https://www.dropbox.com/s/iesj1vpu440mju1/'
                     'ResNet-101-model.caffemodel?dl=1',
        'resnet152': 'https://www.dropbox.com/s/fhmfc4x741iff6i/'
                     'ResNet-152-model.caffemodel?dl=1',
    }

    def __init__(self, n_layers):
        root = download.get_dataset_directory('pfnet/chainer/models/')
        caffemodel_path = os.path.join(
            root, 'ResNet-{}-model.caffemodel'.format(n_layers))
        if not os.path.exists(caffemodel_path):
            if n_layers == 50:
                cache_path = download.cached_download(self.URLS['resnet50'])
            elif n_layers == 101:
                cache_path = download.cached_download(self.URLS['resnet101'])
            elif n_layers == 152:
                cache_path = download.cached_download(self.URLS['resnet152'])
            shutil.move(cache_path, caffemodel_path)
        super(ResNet, self).__init__(
            os.path.basename(caffemodel_path), n_layers=n_layers)

        self._children.remove('fc6')
        del self.fc6
        del self.functions['fc6']
        del self.functions['prob']
        self.train = True

    def __call__(self, x):
        return super(ResNet, self).__call__(
            x, ['res5'], not self.train)['res5']


if __name__ == '__main__':
    model = ResNet(50)
    model.train = False
    import numpy as np
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = model(x)
    print(y.shape)
