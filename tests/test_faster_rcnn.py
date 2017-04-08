#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

import time
import unittest

import numpy as np

import chainer
import cupy as cp
from chainer import computational_graph as cg
from chainer import Variable
from chainer import optimizers
from chainer import testing
from datasets.pascal_voc_dataset import VOC
from models.faster_rcnn import FasterRCNN
from models.vgg16 import VGG16
from models.vgg16 import VGG16Prev


@testing.parameterize(*testing.product({
    'trunk': [VGG16Prev, VGG16],
    'train': [(True, False), (False, True), (False, False)],
    'device': [0],
}))
class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        chainer.set_debug(True)
        np.random.seed(0)
        dataset = VOC('train')
        img, im_info, bbox = dataset[1]
        self.x = Variable(img[None, ...])
        self.im_info = Variable(im_info[None, ...])
        self.gt_boxes = Variable(bbox[None, ...])

    def test_forward_whole(self):
        rpn_in_ch = 512
        rpn_out_ch = 512
        feat_stride = 16
        anchor_ratios = [0.5, 1, 2]
        anchor_scales = [8, 16, 32]
        num_classes = 21
        model = FasterRCNN(
            self.trunk, rpn_in_ch, rpn_out_ch, feat_stride,
            anchor_ratios, anchor_scales, num_classes)
        model.rpn_train, model.rcnn_train = self.train
        if self.device >= 0:
            model.to_gpu(self.device)
            self.x.to_gpu(self.device)
            self.x.volatile = True
            self.assertIs(model.xp, cp)
            self.assertIs(model.trunk.xp, cp)
        st = time.time()
        ret = model(self.x, self.im_info)
        print('Forward whole device:{}, ({}, train:{}): {} sec'.format(
            self.device, self.trunk.__name__, self.train, time.time() - st))
        assert(len(ret) == 2)
        assert(isinstance(ret[0], chainer.Variable))
        assert(isinstance(ret[1], (cp.ndarray, np.ndarray)))

    def test_backward(self):
        rpn_in_ch = 512
        rpn_out_ch = 512
        feat_stride = 16
        anchor_ratios = [0.5, 1, 2]
        anchor_scales = [8, 16, 32]
        num_classes = 21
        model = FasterRCNN(
            self.trunk, rpn_in_ch, rpn_out_ch, feat_stride,
            anchor_ratios, anchor_scales, num_classes)
        model.rpn_train, model.rcnn_train = self.train
        if self.device >= 0:
            model.to_gpu(self.device)
            self.x.to_gpu(self.device)
            self.im_info.to_gpu(self.device)
            self.gt_boxes.to_gpu(self.device)
            self.assertIs(model.xp, cp)
            self.assertIs(model.trunk.xp, cp)
        opt = optimizers.Adam()
        opt.setup(model)

        if model.rpn_train:
            st = time.time()
            rpn_loss = model(self.x, self.im_info, self.gt_boxes)
            model.cleargrads()
            rpn_loss.backward()
            opt.update()
            print('Backward rpn device:{}, ({}, train:{}): {} sec'.format(
                self.device, self.trunk.__name__, self.train, time.time() - st))

            rpn_cg = cg.build_computational_graph([rpn_loss])
            with open('tests/rpn_cg.dot', 'w') as fp:
                fp.write(rpn_cg.dump())

        elif model.rcnn_train:
            st = time.time()
            loss_rcnn = model(self.x, self.im_info, self.gt_boxes)
            model.cleargrads()
            loss_rcnn.backward()
            opt.update()
            print('Backward rcnn device:{}, ({}, train:{}): {} sec'.format(
                self.device, self.trunk.__name__, self.train, time.time() - st))

            loss_rcnn_cg = cg.build_computational_graph([loss_rcnn])
            with open('tests/loss_rcnn_cg.dot', 'w') as fp:
                fp.write(loss_rcnn_cg.dump())


if __name__ == '__main__':
    unittest.main()
