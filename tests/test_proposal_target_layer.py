#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import time
import numpy as np
import cupy as cp
import cv2 as cv
from models.proposal_target_layer import ProposalTargetLayer
from models.region_proposal_network import RegionProposalNetwork
from models.bbox_transform import keep_inside


class TestProposalTargetLayer(unittest.TestCase):

    def setUp(self):
        self.img_info = [224, 224]
        self.gt_boxes = np.array([
            [10, 10, 60, 200, 1],
            [50, 100, 210, 210, 2],
            [160, 40, 200, 70, 3]
        ])
        proposals = np.random.randint(-10, 10, size=(300, 4))
        ext_gt_boxes = self.gt_boxes[:, :4][np.random.randint(3, size=300)]
        self.proposals = ext_gt_boxes + proposals
        self.anchor_target_layer = ProposalTargetLayer()

    def test_call(self):
        self.anchor_target_layer(self.proposals, self.gt_boxes, self.img_info)
