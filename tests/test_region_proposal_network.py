#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

from chainer.cuda import cupy as cp
from models.region_proposal_network import RegionProposalNetwork


class TestRegionProposalNetwork(unittest.TestCase):

    def setUp(self):
        self.rpn = RegionProposalNetwork()
        self.img_info = [600, 800]
        self.feat_h, self.feat_w = map(int, np.array(self.img_info) / 16)

    def test_cpu(self):
        x = np.zeros((1, 512, self.feat_h, self.feat_w), dtype=np.float32)
        self.rpn.train = False
        st = time.time()
        self.rpn(x, self.img_info)
        print(time.time() - st, 'sec')

    def test_gpu(self):
        x = cp.zeros((1, 512, self.feat_h, self.feat_w), dtype=np.float32)
        self.rpn.train = False
        self.rpn.to_gpu(0)
        st = time.time()
        self.rpn(x, self.img_info)
        print(time.time() - st, 'sec')

    def test_forward(self):
        x = np.zeros((1, 512, self.feat_h, self.feat_w), dtype=np.float32)
        self.rpn.train = False
        rois, probs = self.rpn(x, self.img_info)
        print(rois.shape)
        print(probs.shape)
