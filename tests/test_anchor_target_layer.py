#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import time
import numpy as np
import cupy as cp
import cv2 as cv
from models.anchor_target_layer import AnchorTargetLayer
from models.bbox_transform import keep_inside


class TestAnchorTargetLayer(unittest.TestCase):

    def setUp(self):
        self.x = np.arange(256 * 14 * 14, dtype=np.float32).reshape(256, 14, 14)
        self.img_info = [224, 224]
        self.anchor_target_layer = AnchorTargetLayer(16, [0.5, 1, 2], [8, 16, 32])
        self.gt_boxes = np.array([
            [10, 10, 60, 200, 0],
            [50, 100, 210, 210, 1],
            [160, 40, 200, 70, 2]
        ])

    def test_gt_boxes(self):
        gt_canvas = np.zeros((224, 224))
        for gt in self.gt_boxes:
            cv.rectangle(gt_canvas, (gt[0], gt[1]), (gt[2], gt[3]), 255)
        cv.imwrite('tests/gt_boxes.png', gt_canvas)

    def test_inside_anchors(self):
        all_anchors = self.anchor_target_layer._generate_all_anchors(self.x)
        inds_inside, all_inside_anchors = keep_inside(all_anchors, self.img_info)
        anchor_canvas = np.zeros((224, 224))
        for anchor in all_inside_anchors:
            anchor = [int(a) for a in anchor]
            cv.rectangle(anchor_canvas, (anchor[0], anchor[1]),
                         (anchor[2], anchor[3]), 255)
        cv.imwrite('tests/inside_anchors.png', anchor_canvas)

    def test_time(self):
        st = time.time()
        self.anchor_target_layer(self.x, self.gt_boxes, self.img_info)
        print('CPU:', time.time() - st, 'sec')
        x = cp.asarray(self.x)
        gt_boxes = cp.asarray(self.gt_boxes)
        st = time.time()
        self.anchor_target_layer(x, gt_boxes, self.img_info)
        print('GPU:', time.time() - st, 'sec')

    def test_labels(self):
        bbox_labels, bbox_reg_targets = self.anchor_target_layer(self.x, self.gt_boxes, self.img_info)
        all_anchors = self.anchor_target_layer._generate_all_anchors(self.x)

        self.assertEqual(len(bbox_labels), len(all_anchors))
        self.assertEqual(len(bbox_reg_targets), len(all_anchors))

        inds_inside, all_inside_anchors = keep_inside(all_anchors, self.img_info)

        neg_canvas = np.zeros((224, 224))
        pos_canvas = np.zeros((224, 224))
        ign_canvas = np.zeros((224, 224))
        for ind, anchor in zip(inds_inside, all_inside_anchors):
            a = [int(a) for a in anchor]
            self.assertIn(bbox_labels[ind], [-1, 0, 1])
            # Negative
            if bbox_labels[ind] == 0:
                cv.rectangle(neg_canvas, (a[0], a[1]), (a[2], a[3]), 255)
            # Positive
            elif bbox_labels[ind] == 1:
                cv.rectangle(pos_canvas, (a[0], a[1]), (a[2], a[3]), 255)
            # Ignore
            elif bbox_labels[ind] == -1:
                cv.rectangle(ign_canvas, (a[0], a[1]), (a[2], a[3]), 255)
        cv.imwrite('tests/neg_canvas.png', neg_canvas)
        cv.imwrite('tests/pos_canvas.png', pos_canvas)
        cv.imwrite('tests/ign_canvas.png', ign_canvas)

        print(bbox_reg_targets)
