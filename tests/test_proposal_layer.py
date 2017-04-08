#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

import chainer
from chainer import Variable
from models.proposal_layer import ProposalLayer


class TestProposalLayer(unittest.TestCase):

    def setUp(self):
        self.feat_stride = 16
        self.anchor_scales = [8, 16, 32]

    def test_cpu(self):
        st = time.time()
        proposal_layer = ProposalLayer()
        n_anchors = proposal_layer._num_anchors
        for _ in range(10):
            rpn_cls_prob = Variable(
                np.random.rand(1, 2 * n_anchors, 14, 14).astype(np.float32))
            rpn_bbox_pred = Variable(
                np.random.rand(1, 4 * n_anchors, 14, 14).astype(np.float32))
            img_info = Variable(np.array([[224, 224]], np.int32))
            rois, probs = proposal_layer(rpn_cls_prob, rpn_bbox_pred, img_info)
            print('rois:', rois.shape, 'probs:', probs.shape)
        print('cpu mode:', (time.time() - st) / 10., 'sec')

    def test_gpu(self):
        st = time.time()
        cp = chainer.cuda.cupy
        proposal_layer = ProposalLayer()
        n_anchors = proposal_layer._num_anchors
        for _ in range(10):
            rpn_cls_prob = Variable(
                cp.random.rand(1, 2 * n_anchors, 14, 14).astype(cp.float32))
            rpn_bbox_pred = Variable(
                cp.random.rand(1, 4 * n_anchors, 14, 14).astype(cp.float32))
            img_info = Variable(np.array([[224, 224]]))
            rois, probs = proposal_layer(rpn_cls_prob, rpn_bbox_pred, img_info)
            print('rois:', rois.shape, 'probs:', probs.shape)
        print('gpu mode:', (time.time() - st) / 10., 'sec')
