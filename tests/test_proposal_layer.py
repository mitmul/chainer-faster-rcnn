#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from models.proposal_layer import ProposalLayer
import unittest
import numpy as np
import chainer


class TestProposalLayer(unittest.TestCase):

    def setUp(self):
        self.feat_stride = 16
        self.anchor_scales = [4, 8, 16, 32]

    def test_cpu(self):
        st = time.time()
        proposal_layer = ProposalLayer()
        n_anchors = proposal_layer._num_anchors
        rpn_cls_prob = np.random.rand(2 * n_anchors, 14, 14).astype(np.float32)
        rpn_bbox_pred = np.random.rand(4 * n_anchors, 14, 14).astype(np.float32)
        img_info = [640, 480, 0.5]
        rois = proposal_layer(rpn_cls_prob, rpn_bbox_pred, img_info)
        print(rois)
        print(time.time() - st, 'sec')

    def test_gpu(self):
        st = time.time()
        cp = chainer.cuda.cupy
        proposal_layer = ProposalLayer(xp=cp)
        n_anchors = proposal_layer._num_anchors
        rpn_cls_prob = cp.random.rand(2 * n_anchors, 14, 14).astype(cp.float32)
        rpn_bbox_pred = cp.random.rand(4 * n_anchors, 14, 14).astype(cp.float32)
        img_info = [640, 480, 0.5]
        rois = proposal_layer(rpn_cls_prob, rpn_bbox_pred, img_info)
        print(rois)
        print(time.time() - st, 'sec')
