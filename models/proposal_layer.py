#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# Original work by:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# https://github.com/rbgirshick/py-faster-rcnn
# -----------------------------------------------------------------------------

import numpy

import chainer
from models.bbox_transform import bbox_transform_inv
from models.bbox_transform import clip_boxes
from models.cpu_nms import cpu_nms
from models.generate_anchors import generate_anchors
from models.gpu_nms import gpu_nms


class ProposalLayer(object):
    """Generate deterministic proposal regions

    Calculate object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").

    Args:
        feat_stride (int): The stride of corresponding pixels in the lowest
            layer (image). A couple of adjacent values on the input feature map
            are actually distant from each other with `feat_stride` pixels in
            the image plane.
        anchor_scales (list): A list of scales of anchor boxes. See
    """
    RPN_NMS_THRESH = 0.7
    TRAIN_RPN_PRE_NMS_TOP_N = 12000
    TRAIN_RPN_POST_NMS_TOP_N = 2000
    TEST_RPN_PRE_NMS_TOP_N = 6000
    TEST_RPN_POST_NMS_TOP_N = 300
    RPN_MIN_SIZE = 16

    def __init__(self, feat_stride=16, anchor_scales=[4, 8, 16, 32], xp=numpy):
        self._feat_stride = feat_stride
        self._anchors = xp.asarray(generate_anchors(
            scales=numpy.array(anchor_scales), xp=xp))
        self._num_anchors = len(self._anchors)
        self._pre_nms_top_n = self.TRAIN_RPN_PRE_NMS_TOP_N
        self._post_nms_top_n = self.TRAIN_RPN_POST_NMS_TOP_N
        self._nms_thresh = self.RPN_NMS_THRESH
        self._min_size = self.RPN_MIN_SIZE
        self._train = True
        self.xp = xp

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, value):
        self._train = value
        if value:
            self._pre_nms_top_n = self.TRAIN_RPN_PRE_NMS_TOP_N
            self._post_nms_top_n = self.TRAIN_RPN_POST_NMS_TOP_N
        else:
            self._pre_nms_top_n = self.TEST_RPN_PRE_NMS_TOP_N
            self._post_nms_top_n = self.TEST_RPN_POST_NMS_TOP_N

    def _generate_anchors(self, feat_h, feat_w):
        # Create lattice
        shift_x = self.xp.arange(0, feat_w) * self._feat_stride
        shift_y = self.xp.arange(0, feat_h) * self._feat_stride
        shift_x, shift_y = self.xp.meshgrid(shift_x, shift_y)
        shifts = self.xp.vstack((shift_x.ravel(), shift_y.ravel(),
                                 shift_x.ravel(), shift_y.ravel())).transpose()

        # Create all shifted anchors
        A = self._num_anchors
        K = len(shifts)  # number of lattice points = feat_h * feat_w
        anchors = self._anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
        anchors = anchors.reshape(K * A, 4)
        return anchors

    def __call__(self, rpn_cls_prob, rpn_bbox_pred, img_info):
        """It takes numpy or cupy arrays

        Args:
            rpn_cls_prob (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(2 * n_anchors, feat_h, feat_w)`-shaped array.
            rpn_bbox_pred (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(4 * n_anchors, feat_h, feat_w)`-shaped array.
            img_info (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(3,)`-shaped list that contains
                :math:`(img_h, img_w, img_scale)`.
        """

        _, feat_h, feat_w = rpn_bbox_pred.shape
        anchors = self._generate_anchors(feat_h, feat_w)

        # Predicted anchor transformation and class probability
        bbox_deltas = rpn_bbox_pred.transpose(1, 2, 0).reshape(-1, 4)

        # Apply the transformation to the base anchors
        proposals = bbox_transform_inv(anchors, bbox_deltas, -1)

        # Clip predicted boxes to image
        proposals = clip_boxes(proposals, img_info[:2])

        # Remove predicted boxes with either height or width < threshold
        keep = _filter_boxes(proposals, self._min_size, self.xp)
        proposals = proposals[keep]

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        fg_probs = rpn_cls_prob[self._num_anchors:]
        fg_probs = fg_probs.transpose(1, 2, 0).reshape(-1, 1)
        fg_probs = fg_probs[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest and
        # take top pre_nms_topN (e.g. 6000)
        order = fg_probs.ravel()
        order = chainer.cuda.to_cpu(order).argsort()[::-1]
        order = self.xp.asarray(order)
        if self._pre_nms_top_n > 0:
            order = order[:self._pre_nms_top_n]
        proposals = proposals[order]
        fg_probs = fg_probs[order]

        # Apply nms (e.g. threshold = 0.7)
        # Take after_nms_top_n (e.g. 300)
        # return the top proposals (-> RoIs top)
        if isinstance(fg_probs, numpy.ndarray):
            keep = cpu_nms(self.xp.hstack((proposals, fg_probs)), self._nms_thresh)
        elif isinstance(fg_probs, chainer.cuda.cupy.ndarray):
            # TODO(mitmul): Current gpu_nms assumes the input array is on CPU
            # and it performs memory allocation and transfer inside the CDUA
            # kernel. It should be replaced with CuPy implementation.
            fg_probs = chainer.cuda.to_cpu(fg_probs)
            proposals = chainer.cuda.to_cpu(proposals)
            keep = gpu_nms(numpy.hstack((proposals, fg_probs)), self._nms_thresh)
        if self._post_nms_top_n > 0:
            keep = keep[:self._post_nms_top_n]
        proposals = proposals[keep]
        fg_probs = fg_probs[keep]

        return proposals


def _filter_boxes(boxes, min_size, xp=numpy):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = xp.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
