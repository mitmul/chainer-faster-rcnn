#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from lib.models.FasterRCNN import FasterRCNN
from lib.models.VGG16 import VGG16

import chainer
import unittest


class TestFasterRCNN(unittest.TestCase):

    def setUp(self):
        pass

    def test_cpu_VGG16(self):
        gpu = -1
        trunk = VGG16
        n_anchors = 9
        feat_stride = 16
        anchor_scales = [8, 16, 32]
        num_classes = 2
        model = FasterRCNN(gpu, trunk, n_anchors, feat_stride,
                           anchor_scales, num_classes)
