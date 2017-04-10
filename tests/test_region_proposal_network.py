#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

from chainer import Variable
from chainer.cuda import cupy as cp
from models.region_proposal_network import RegionProposalNetwork


class TestRegionProposalNetwork(unittest.TestCase):

    def setUp(self):
        self.rpn = RegionProposalNetwork()
        self.img_info = Variable(np.array([[600, 800]]))
        self.feat_h, self.feat_w = 600 // 16, 800 // 16

    def test_cpu(self):
        x = Variable(np.zeros(
            (1, 512, self.feat_h, self.feat_w), dtype=np.float32))
        self.rpn.train = False
        st = time.time()
        self.rpn(x, self.img_info)
        print(time.time() - st, 'sec')

    def test_gpu(self):
        x = Variable(cp.zeros(
            (1, 512, self.feat_h, self.feat_w), dtype=cp.float32))
        self.rpn.train = False
        self.rpn.to_gpu(0)
        st = time.time()
        self.rpn(x, self.img_info)
        print(time.time() - st, 'sec')

    def test_forward(self):
        x = Variable(np.zeros(
            (1, 512, self.feat_h, self.feat_w), dtype=np.float32))
        self.rpn.train = False
        rois, probs = self.rpn(x, self.img_info)
        print(rois.shape)
        print(probs.shape)
