#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 Shunta Saito

from lib.faster_rcnn.bbox_transform import bbox_transform_inv
from lib.faster_rcnn.bbox_transform import clip_boxes
from lib.faster_rcnn.proposal_layer import ProposalLayer

import chainer
import chainer.functions as F
import chainer.links as L


class VGG16(chainer.Chain):

    def __init__(self, train=False):
        super(VGG16, self).__init__()
        self.trunk = [
            ('conv1_1', L.Convolution2D(3, 64, 3, 1, 1)),
            ('relu1_1', F.ReLU()),
            ('conv1_2', L.Convolution2D(64, 64, 3, 1, 1)),
            ('relu1_2', F.ReLU()),
            ('pool1', F.MaxPooling2D(2, 2)),
            ('conv2_1', L.Convolution2D(64, 128, 3, 1, 1)),
            ('relu2_1', F.ReLU()),
            ('conv2_2', L.Convolution2D(128, 128, 3, 1, 1)),
            ('relu2_2', F.ReLU()),
            ('pool2', F.MaxPooling2D(2, 2)),
            ('conv3_1', L.Convolution2D(128, 256, 3, 1, 1)),
            ('relu3_1', F.ReLU()),
            ('conv3_2', L.Convolution2D(256, 256, 3, 1, 1)),
            ('relu3_2', F.ReLU()),
            ('conv3_3', L.Convolution2D(256, 256, 3, 1, 1)),
            ('relu3_3', F.ReLU()),
            ('pool3', F.MaxPooling2D(2, 2)),
            ('conv4_1', L.Convolution2D(256, 512, 3, 1, 1)),
            ('relu4_1', F.ReLU()),
            ('conv4_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu4_2', F.ReLU()),
            ('conv4_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu4_3', F.ReLU()),
            ('pool4', F.MaxPooling2D(2, 2)),
            ('conv5_1', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_1', F.ReLU()),
            ('conv5_2', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_2', F.ReLU()),
            ('conv5_3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('relu5_3', F.ReLU()),
            ('rpn_conv_3x3', L.Convolution2D(512, 512, 3, 1, 1)),
            ('rpn_relu_3x3', F.ReLU()),
        ]
        for name, link in self.trunk:
            if 'conv' in name:
                self.add_link(name, link)

    def __call__(self, x):
        for name, f in self.trunk:
            x = (getattr(self, name) if 'conv' in name else f)(x)
            if 'relu5_3' in name:
                self.relu5_3_out = x
        return x


class FasterRCNN(chainer.Chain):

    def __init__(self, trunk=VGG16):
        super(FasterRCNN, self).__init__()
        self.add_link('trunk', trunk())
        self.add_link('rpn_cls_score', L.Convolution2D(512, 18, 1, 1, 0))
        self.add_link('rpn_bbox_pred', L.Convolution2D(512, 36, 1, 1, 0))
        self.add_link('fc6', L.Linear(25088, 4096))
        self.add_link('fc7', L.Linear(4096, 4096))
        self.add_link('cls_score', L.Linear(4096, 21))
        self.add_link('bbox_pred', L.Linear(4096, 84))
        self.train = True

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
        boxes = rois[:, 1:5] / im_info[0][2]
        rois = chainer.Variable(rois, volatile=not self.train)

        # RCNN
        pool5 = F.roi_pooling_2d(self.trunk.relu5_3_out, rois, 7, 7, 0.0625)
        fc6 = F.relu(self.fc6(pool5))
        fc7 = F.relu(self.fc7(fc6))
        self.scores = F.softmax(self.cls_score(fc7))

        box_deltas = self.bbox_pred(fc7).data
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        self.pred_boxes = clip_boxes(pred_boxes, im_info[0][:2])

        if self.train:
            # loss_cls = F.softmax_cross_entropy(cls_score, labels)
            # huber loss with delta=1 means SmoothL1Loss
            return None
        else:
            return self.scores, self.pred_boxes
