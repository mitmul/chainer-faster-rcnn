#!/usr/bin/env python
# -*- coding: utf-8 -*-

from lib.models.vgg16 import VGG16

import chainer
import imp
import numpy as np

if __name__ == '__main__':
    model_file = 'lib/models/faster_rcnn.py'
    model_name = 'FasterRCNN'
    train = True
    trunk_path = 'data/VGG16.model'
    trunk = np.load(trunk_path)

    model = imp.load_source(model_name, model_file)
    model = getattr(model, model_name)
    model = model(
        gpu=-1,
        trunk=VGG16,
        rpn_in_ch=512,
        rpn_out_ch=512,
        n_anchors=9,
        feat_stride=16,
        anchor_scales=[8, 16, 32],
        num_classes=21,
        spatial_scale=0.0625,
        rpn_sigma=1.0,
        sigma=3.0
    )

    # Initialize
    if trunk_path is not None:
        for name, param in model.namedparams():
            if 'trunk' not in name:
                continue
            for trunk_name in trunk.keys():
                if trunk_name in name:
                    target = model.trunk
                    names = [n for n in trunk_name.split('/') if len(n) > 0]
                    for n in names[:-1]:
                        target = target.__dict__[n]
                    print(trunk_name, target.__dict__[names[-1]].data.shape,
                          trunk[trunk_name].shape)
                    target.__dict__[names[-1]].data = trunk[trunk_name]
