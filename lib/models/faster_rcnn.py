#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from chainer.cuda import get_array_module
from chainer.cuda import to_gpu
from lib.faster_rcnn.bbox_transform import bbox_transform_inv
from lib.faster_rcnn.bbox_transform import clip_boxes
from lib.faster_rcnn.proposal_layer import ProposalLayer
from lib.models.VGG16 import VGG16

import chainer
import chainer.functions as F
import chainer.links as L


class FasterRCNN(chainer.Chain):

    def __init__(self, gpu=-1, trunk=VGG16):
        super(FasterRCNN, self).__init__()
        self.add_link('trunk', trunk())
        self.add_link('rpn_cls_score', L.Convolution2D(512, 18, 1, 1, 0))
        self.add_link('rpn_bbox_pred', L.Convolution2D(512, 36, 1, 1, 0))
        self.add_link('fc6', L.Linear(25088, 4096))
        self.add_link('fc7', L.Linear(4096, 4096))
        self.add_link('cls_score', L.Linear(4096, 21))
        self.add_link('bbox_pred', L.Linear(4096, 84))
        self.train = True
        self.gpu = gpu

        self.proposal_layer = ProposalLayer(
            feat_stride=16, anchor_scales=[8, 16, 32])

    def __call__(self, x, im_info):
        h, n = self.trunk(x), x.data.shape[0]
        rpn_cls_score = self.rpn_cls_score(h)
        c, hh, ww = rpn_cls_score.data.shape[1:]
        rpn_bbox_pred = self.rpn_bbox_pred(h)
        rpn_cls_score = F.reshape(rpn_cls_score, (n, 2, -1))

        # RoI Proposal
        rpn_cls_prob = F.softmax(rpn_cls_score)
        rpn_cls_prob_reshape = F.reshape(rpn_cls_prob, (n, c, hh, ww))
        rois = self.proposal_layer(
            rpn_cls_prob_reshape, rpn_bbox_pred, im_info, self.train)
        if self.gpu >= 0:
            rois = to_gpu(rois, device=self.gpu)
            im_info = to_gpu(im_info, device=self.gpu)
            with chainer.cuda.Device(self.gpu):
                boxes = rois[:, 1:5] / im_info[0][2]
        else:
            boxes = rois[:, 1:5] / im_info[0][2]
        rois = chainer.Variable(rois, volatile=not self.train)

        # RCNN
        pool5 = F.roi_pooling_2d(self.trunk.feature, rois, 7, 7, 0.0625)
        fc6 = F.relu(self.fc6(pool5))
        fc7 = F.relu(self.fc7(fc6))
        self.scores = F.softmax(self.cls_score(fc7))

        box_deltas = self.bbox_pred(fc7).data
        pred_boxes = bbox_transform_inv(boxes, box_deltas, self.gpu)
        self.pred_boxes = clip_boxes(pred_boxes, im_info[0][:2], self.gpu)

        if self.train:
            # loss_cls = F.softmax_cross_entropy(cls_score, labels)
            # huber loss with delta=1 means SmoothL1Loss
            return None
        else:
            return self.scores, self.pred_boxes
