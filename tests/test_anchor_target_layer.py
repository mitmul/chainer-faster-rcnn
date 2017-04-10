#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import unittest

import numpy as np

import cupy as cp
import cv2 as cv
from chainer import Variable
from models.anchor_target_layer import AnchorTargetLayer
from models.bbox_transform import keep_inside


class TestAnchorTargetLayer(unittest.TestCase):
    def setUp(self):
        x = np.arange(2 * 9 * 14 * 14, dtype=np.float32)
        self.x = Variable(x.reshape(1, 18, 14, 14))
        self.img_info = Variable(np.array([[224, 224]], dtype=np.int32))
        self.anchor_target_layer = AnchorTargetLayer(
            16, [0.5, 1, 2], [8, 16, 32])
        self.gt_boxes = Variable(np.array([[
            [10, 10, 60, 200, 0],
            [50, 100, 210, 210, 1],
            [160, 40, 200, 70, 2]
        ]], dtype=np.float32))
        self.feat_h, self.feat_w = 14, 14

    def test_gt_boxes(self):
        gt_canvas = np.zeros((224, 224))
        for gt in self.gt_boxes.data[0]:
            cv.rectangle(gt_canvas, (gt[0], gt[1]), (gt[2], gt[3]), 255)
        cv.imwrite('tests/gt_boxes.png', gt_canvas)

    def test_inside_anchors(self):
        all_anchors = \
            self.anchor_target_layer._generate_all_bbox_use_array_info(
                self.x.data[0])
        inds_inside, all_inside_anchors = keep_inside(
            all_anchors, self.img_info.data[0])
        anchor_canvas = np.zeros((224, 224))
        for anchor in all_inside_anchors:
            anchor = [int(a) for a in anchor]
            cv.rectangle(anchor_canvas, (anchor[0], anchor[1]),
                         (anchor[2], anchor[3]), 255)
        cv.imwrite('tests/inside_anchors.png', anchor_canvas)

    def test_time(self):
        st = time.time()
        self.anchor_target_layer(self.feat_h, self.feat_w, self.gt_boxes,
                                 self.img_info)
        print('CPU:', time.time() - st, 'sec')
        self.x.to_gpu(0)
        self.gt_boxes.to_gpu(0)
        self.img_info.to_gpu(0)
        st = time.time()
        self.anchor_target_layer(self.feat_h, self.feat_w, self.gt_boxes,
                                 self.img_info)
        print('GPU:', time.time() - st, 'sec')

    def test_forward(self):
        bbox_labels, bbox_reg_targets, inds_inside, n_all_bbox = \
            self.anchor_target_layer(self.feat_h, self.feat_w, self.gt_boxes,
                                     self.img_info)
        print(bbox_labels.shape, bbox_reg_targets.shape)

    def test_labels(self):
        bbox_labels, bbox_reg_targets, inds_inside, n_all_bbox = \
            self.anchor_target_layer(self.feat_h, self.feat_w, self.gt_boxes,
                                     self.img_info)
        all_anchors = \
            self.anchor_target_layer._generate_all_bbox_use_array_info(
                self.x.data[0])

        self.assertEqual(len(bbox_labels), len(inds_inside))
        self.assertEqual(len(bbox_reg_targets), len(inds_inside))

        inds_inside, all_inside_anchors = keep_inside(
            all_anchors, self.img_info.data[0])

        neg_canvas = np.zeros((224, 224))
        pos_canvas = np.zeros((224, 224))
        ign_canvas = np.zeros((224, 224))
        for lbl, ind, anchor in zip(bbox_labels, inds_inside,
                                    all_inside_anchors):
            a = [int(a) for a in anchor]
            self.assertIn(lbl, [-1, 0, 1])
            # Negative
            if lbl == 0:
                cv.rectangle(neg_canvas, (a[0], a[1]), (a[2], a[3]), 255)
            # Positive
            elif lbl == 1:
                cv.rectangle(pos_canvas, (a[0], a[1]), (a[2], a[3]), 255)
            # Ignore
            elif lbl == -1:
                cv.rectangle(ign_canvas, (a[0], a[1]), (a[2], a[3]), 255)
        cv.imwrite('tests/neg_canvas.png', neg_canvas)
        cv.imwrite('tests/pos_canvas.png', pos_canvas)
        cv.imwrite('tests/ign_canvas.png', ign_canvas)
