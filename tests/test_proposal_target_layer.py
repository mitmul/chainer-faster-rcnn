#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

import cupy as cp
import cv2 as cv
from chainer import Variable
from models.bbox_transform import keep_inside
from models.proposal_target_layer import ProposalTargetLayer
from models.region_proposal_network import RegionProposalNetwork


class TestProposalTargetLayer(unittest.TestCase):

    def setUp(self):
        self.img_info = [224, 224]
        gt_boxes = np.array([
            [10, 10, 60, 200, 1],
            [50, 100, 210, 210, 2],
            [160, 40, 200, 70, 3]
        ], dtype=np.float32)
        jitter = np.random.randint(-10, 10, size=(300, 4))
        ext_gt_boxes = gt_boxes[np.random.randint(3, size=300), :4]
        proposals = (ext_gt_boxes + jitter).astype(np.float32)
        self.proposals = proposals
        self.gt_boxes = Variable(gt_boxes[None, ...])
        self.proposal_target_layer = ProposalTargetLayer()

    def test_call(self):
        self.proposal_target_layer(self.proposals, self.gt_boxes)
