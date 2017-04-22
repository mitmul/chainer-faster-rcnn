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

import os

import numpy as np

from chainer import Variable
from chainer import cuda
from models.bbox_transform import bbox_transform_inv
from models.bbox_transform import clip_boxes
from models.bbox_transform import filter_boxes
from models.cpu_nms import cpu_nms
from models.generate_anchors import generate_anchors
from models.gpu_nms import gpu_nms


class ProposalLayer(object):
    """Generate proposal regions

    Converts RPN outputs (per-anchor scores and bbox regression estimates) into
    object proposals.

    It calculates object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors"). All proposals
    are at the scale of the input image size given by `img_info` to the
    `__call__` method.

    Args:
        feat_stride (int): The stride of corresponding pixels in the lowest
            layer (image plane). A couple of adjacent values on the input
            feature map are actually distant from each other with `feat_stride`
            pixels in the image plane. It depends on the trunk model
            architecture, e.g., 16 for conv5_3 of VGG16.
        anchor_scales (list of integers): A list of scales of anchor boxes.

    """

    RPN_NMS_THRESH = 0.7
    TRAIN_RPN_PRE_NMS_TOP_N = 12000
    TRAIN_RPN_POST_NMS_TOP_N = 2000
    TEST_RPN_PRE_NMS_TOP_N = 6000
    TEST_RPN_POST_NMS_TOP_N = 300
    RPN_MIN_SIZE = 16

    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __init__(self, feat_stride=16, anchor_ratios=(0.5, 1, 2),
                 anchor_scales=(8, 16, 32)):
        self._feat_stride = feat_stride
        self._anchors = generate_anchors(
            ratios=anchor_ratios, scales=anchor_scales)
        self._num_anchors = len(self._anchors)  # Typically 9
        self._nms_thresh = np.float(self.RPN_NMS_THRESH)
        self._min_size = self.RPN_MIN_SIZE
        self._train = True
        self.train = self._train  # To set _pre_nms_top_n and _post_nms_top_n

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

    def _check_data_type_forward(self, rpn_cls_prob, rpn_bbox_pred, img_info):
        assert rpn_cls_prob.shape[0] == 1
        assert rpn_cls_prob.shape[1] == 2 * self._num_anchors
        assert rpn_cls_prob.ndim == 4
        assert rpn_cls_prob.dtype.kind == 'f'
        assert isinstance(rpn_cls_prob, Variable)

        assert rpn_bbox_pred.shape[0] == 1
        assert rpn_bbox_pred.shape[1] == 4 * self._num_anchors
        assert rpn_bbox_pred.ndim == 4
        assert rpn_bbox_pred.dtype.kind == 'f'
        assert isinstance(rpn_bbox_pred, Variable)

        assert img_info.shape == (1, 2)
        assert img_info.dtype.kind == 'i'
        assert isinstance(img_info, Variable)

    def __call__(self, rpn_cls_prob, rpn_bbox_pred, img_info):
        """It takes numpy or cupy arrays

        Args:
            rpn_cls_prob (:class:`~chainer.Variable`):
                :math:`(1, 2 * n_anchors, feat_h, feat_w)`-shaped array.
            rpn_bbox_pred (:class:`~chainer.Variable`):
                :math:`(1, 4 * n_anchors, feat_h, feat_w)`-shaped array.
            img_info (:class:`~chainer.Variable`): The input image size
                represented as a list of integers such as
                :math:`(img_h, img_w)`. And the batchsize should be 1, so the
                shape should be :math:`(1, 2)`.

        Returns:
            proposals (:class:`~chainer.Variable`):
                A set of proposal rectangles represented in
                :math:`(x_min, x_max, y_min, y_max)`. The scale of values are
                at the input image size given by `img_info` argument.

        """
        if self.type_check_enable:
            self._check_data_type_forward(
                rpn_cls_prob, rpn_bbox_pred, img_info)

        with cuda.get_device_from_array(rpn_cls_prob.data) as d:
            xp = cuda.get_array_module(rpn_cls_prob)

            # Currently it assumes that the batchsize is always 1
            rpn_cls_prob = rpn_cls_prob.data[0]
            rpn_bbox_pred = rpn_bbox_pred.data[0]
            img_info = img_info.data[0]

            # Generate all regions whose scale is at the input image plane
            all_bbox = self._generate_all_bbox_use_array_info(rpn_bbox_pred)

            # Reshape anchor transformation to (A * feat_h * feat_w, 4)
            bbox_trans = rpn_bbox_pred.transpose(1, 2, 0).reshape(-1, 4)

            # Apply the transformation to the base anchors
            proposals = bbox_transform_inv(all_bbox, bbox_trans)

            # Clip predicted boxes to image
            proposals = clip_boxes(proposals, img_info)

            # Remove predicted boxes with either height or width < threshold
            keep = filter_boxes(proposals, self._min_size)
            proposals = proposals[keep]

            # the first set of _num_anchors channels are bg probs
            # the second set are the fg probs
            fg_probs = rpn_cls_prob[self._num_anchors:]
            fg_probs = fg_probs.transpose(1, 2, 0).reshape(-1, 1)
            fg_probs = fg_probs[keep]

            # Sort all (proposal, score) pairs by score from highest to lowest and
            # take top pre_nms_topN (e.g. 6000)
            order = fg_probs.ravel()

            # TODO(mitmul): User cupy.argsort when it becomes available
            if isinstance(order, cuda.cupy.ndarray):
                order = cuda.to_cpu(order).argsort()[::-1]
                order = xp.asarray(order)
            else:
                order = order.argsort()[::-1]

            if self._pre_nms_top_n > 0:
                order = order[:self._pre_nms_top_n]
            proposals = proposals[order]
            fg_probs = fg_probs[order]

            # Apply nms (e.g. threshold = 0.7)
            # Take after_nms_top_n (e.g. 300)
            # return the top proposals (-> RoIs top)
            # TODO(mitmul): Fix this workaround for GPU memory limit
            proposals = cuda.to_cpu(proposals)
            fg_probs = cuda.to_cpu(fg_probs)
            keep = cpu_nms(np.hstack((proposals, fg_probs)), self._nms_thresh)

            # if xp is not cuda.cupy:
            #     keep = cpu_nms(np.hstack((proposals, fg_probs)), self._nms_thresh)
            # else:
            #     # TODO(mitmul): Improve gpu_nms to take GPU array directly
            #     dets = cuda.to_cpu(xp.hstack((proposals, fg_probs))).astype(
            #         np.float32)
            #     keep = gpu_nms(dets, self._nms_thresh)
            #     keep = xp.asarray(keep)

            if self._post_nms_top_n > 0:
                keep = keep[:self._post_nms_top_n]

            proposals = proposals[keep]
            fg_probs = fg_probs[keep]

            proposals = xp.asarray(proposals)
            fg_probs = xp.asarray(fg_probs)

            return proposals, fg_probs

    def _generate_all_bbox_use_array_info(self, rpn_bbox_pred):
        with cuda.get_device_from_array(rpn_bbox_pred) as d:
            xp = cuda.get_array_module(rpn_bbox_pred)
            _, feat_h, feat_w = rpn_bbox_pred.shape
            return xp.asarray(self._generate_all_bbox(feat_h, feat_w),
                              dtype=xp.float32)

    def _generate_all_bbox(self, feat_h, feat_w):
        # Create lattice (base points to shift anchors)
        shift_x = np.arange(0, feat_w) * self._feat_stride
        shift_y = np.arange(0, feat_h) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Create all bbox
        A = self._num_anchors
        K = len(shifts)  # number of base points = feat_h * feat_w

        bbox = self._anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
        bbox = bbox.reshape(K * A, 4)
        return bbox
