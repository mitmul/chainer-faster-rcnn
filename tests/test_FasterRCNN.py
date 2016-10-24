#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer import optimizers
from lib.models.faster_rcnn import FasterRCNN
from lib.models.vgg16 import VGG16

import chainer
import numpy as np
import unittest


class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        chainer.set_debug(True)
        np.random.seed(0)
        x = np.random.randint(0, 255, size=(224, 224, 3)).astype(np.float)
        x -= np.array([[[102.9801, 115.9465, 122.7717]]])
        self.x = np.expand_dims(x, 0).transpose(0, 3, 1, 2).astype(np.float32)
        self.im_info = np.array([[224, 224, 1.6]])
        self.gt_boxes = np.array([
            [10, 10, 60, 200, 0],
            [50, 100, 210, 210, 1],
            [160, 40, 200, 70, 2]
        ])

    def test_forward_cpu_VGG16(self):
        print('test_forward_cpu_VGG16')
        gpu = -1
        trunk = VGG16
        rpn_in_ch = 512
        rpn_out_ch = 512
        n_anchors = 9
        feat_stride = 16
        anchor_scales = [8, 16, 32]
        num_classes = 21
        spatial_scale = 0.0625
        model = FasterRCNN(
            gpu, trunk, rpn_in_ch, rpn_out_ch, n_anchors, feat_stride,
            anchor_scales, num_classes, spatial_scale)

        model.train = False
        ret = model(chainer.Variable(self.x, volatile=True), self.im_info)
        assert(len(ret) == 2)
        assert(isinstance(ret[0], chainer.Variable))
        assert(isinstance(ret[1], np.ndarray))

    def test_backward_cpu_VGG16(self):
        gpu = -1
        trunk = VGG16
        rpn_in_ch = 512
        rpn_out_ch = 512
        n_anchors = 9
        feat_stride = 16
        anchor_scales = [8, 16, 32]
        num_classes = 21
        spatial_scale = 0.0625
        model = FasterRCNN(
            gpu, trunk, rpn_in_ch, rpn_out_ch, n_anchors, feat_stride,
            anchor_scales, num_classes, spatial_scale)
        opt = optimizers.Adam()
        opt.setup(model)

        model.train = True

        rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls = \
            model(chainer.Variable(self.x, volatile=False),
                  self.im_info, self.gt_boxes)
        model.zerograds()
        rpn_cls_loss.backward()
        opt.update()

        rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls = \
            model(chainer.Variable(self.x, volatile=False),
                  self.im_info, self.gt_boxes)
        model.zerograds()
        rpn_loss_bbox.backward()
        opt.update()

        rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls = \
            model(chainer.Variable(self.x, volatile=False),
                  self.im_info, self.gt_boxes)
        model.zerograds()
        loss_bbox.backward()
        opt.update()

        rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls = \
            model(chainer.Variable(self.x, volatile=False),
                  self.im_info, self.gt_boxes)
        model.zerograds()
        loss_cls.backward()
        opt.update()
