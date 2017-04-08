#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# Original work by:
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

import os

import numpy as np
from chainer import Variable
from chainer import cuda

from models.bbox import bbox_overlaps
from models.bbox_transform import bbox_transform
from models.bbox_transform import keep_inside
from models.proposal_layer import ProposalLayer


class AnchorTargetLayer(ProposalLayer):
    """Assign anchors to ground-truth targets

    It produces:
        1. anchor classification labels
        2. bounding-box regression targets.

    Args:
        feat_stride (int): The stride of corresponding pixels in the lowest
            layer (image). A couple of adjacent values on the input feature map
            are actually distant from each other with `feat_stride` pixels in
            the image plane.
        scales (list of integers): A list of scales of anchor boxes.

    """

    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 256

    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __init__(self, feat_stride=16, anchor_ratios=(0.5, 1, 2),
                 anchor_scales=(8, 16, 32)):
        super(AnchorTargetLayer, self).__init__(
            feat_stride, anchor_ratios, anchor_scales)

    def _check_data_type_forward(self, rpn_cls_prob, gt_boxes, img_info):
        assert rpn_cls_prob.shape[0] == 1
        assert rpn_cls_prob.shape[1] == 2 * self._num_anchors
        assert rpn_cls_prob.ndim == 4
        assert rpn_cls_prob.dtype.kind == 'f'
        assert isinstance(rpn_cls_prob, Variable)

        assert gt_boxes.shape[0] == 1
        assert gt_boxes.shape[2] == 5
        assert gt_boxes.dtype.kind == 'f'
        assert isinstance(gt_boxes, Variable)

        assert img_info.shape == (1, 2)
        assert img_info.dtype.kind == 'i'
        assert isinstance(img_info, Variable)

    def __call__(self, rpn_cls_prob, gt_boxes, img_info):
        """It takes numpy or cupy arrays

        Args:
            rpn_cls_prob (:class:`~chainer.Variable`):
                :math:`(1, 2 * n_anchors, feat_h, feat_w)`-shaped array.
            gt_boxes (:class:`~chainer.Variable`): The ground truth bounding
                boxes and its class label array. The shape should be
                :math:`(1, n_gt_boxes, 5)` and the batchsize should be 1.
                Each 5-dimensional vector has :math:`(x1, y1, x2, y2, cls_id)`.
                The scale of these values is at the input image scale.
            img_info (:class:`~chainer.Variable`): The input image info. It
                contains :math:`(height, width)` and the batchsize should be 1.
                So the shape should be :math:`(1, 2)`.

        """
        if self.type_check_enable:
            self._check_data_type_forward(rpn_cls_prob, gt_boxes, img_info)

        # Currently it assumes that the batchsize is always 1
        rpn_cls_prob = rpn_cls_prob.data[0]
        gt_boxes = gt_boxes.data[0]
        img_info = img_info.data[0]

        all_anchors = self._generate_all_anchors(rpn_cls_prob)
        inds_inside, all_inside_anchors = keep_inside(all_anchors, img_info)
        argmax_overlaps_inds, bbox_labels = \
            self._create_bbox_labels(inds_inside, all_inside_anchors, gt_boxes)

        # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
        xp = cuda.get_array_module(rpn_cls_prob)
        gt_boxes = gt_boxes[argmax_overlaps_inds]
        bbox_reg_targets = bbox_transform(all_inside_anchors, gt_boxes)
        bbox_labels_out = xp.ones((all_anchors.shape[0],), dtype=xp.int32) * -1
        bbox_labels_out[inds_inside] = bbox_labels
        bbox_reg_targets_out = xp.ones_like(all_anchors, dtype=xp.float32) * -1
        bbox_reg_targets_out[inds_inside, :] = bbox_reg_targets
        return bbox_labels_out, bbox_reg_targets_out

    def _create_bbox_labels(self, inds_inside, anchors, gt_boxes):
        """Create bbox labels.

        label: 1 is positive, 0 is negative, -1 is dont care
        """
        xp = cuda.get_array_module(anchors)

        # assign ignore labels first
        labels = xp.ones((len(inds_inside),), dtype=xp.int32) * -1

        argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = \
            self._calc_overlaps(anchors, gt_boxes, inds_inside)

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps_inds] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)
        fg_inds = xp.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            # TODO(mitmul): Use CuPy when cupy.random.choice becomes available
            fg_inds = cuda.to_cpu(fg_inds)
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - xp.sum(labels == 1)
        bg_inds = xp.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            # TODO(mitmul): Use CuPy when cupy.random.choice becomes available
            bg_inds = cuda.to_cpu(bg_inds)
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        return argmax_overlaps_inds, labels

    def _calc_overlaps(self, anchors, gt_boxes, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        # TODO(mitmul): Implement GPU version of bbox_overlaps
        xp = cuda.get_array_module(anchors)
        anchors = cuda.to_cpu(anchors)
        gt_boxes = cuda.to_cpu(gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        overlaps = xp.asarray(overlaps)

        argmax_overlaps_inds = overlaps.argmax(axis=1)
        gt_argmax_overlaps_inds = overlaps.argmax(axis=0)

        max_overlaps = overlaps[
            xp.arange(len(inds_inside)), argmax_overlaps_inds]
        gt_max_overlaps = overlaps[
            gt_argmax_overlaps_inds, xp.arange(overlaps.shape[1])]
        gt_argmax_overlaps_inds = xp.where(overlaps == gt_max_overlaps)[0]

        argmax_overlaps_inds = xp.asarray(argmax_overlaps_inds)
        max_overlaps = xp.asarray(max_overlaps)
        gt_argmax_overlaps_inds = xp.asarray(gt_argmax_overlaps_inds)

        return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds
