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

    def _check_data_type_forward(self, gt_boxes, img_info):
        assert gt_boxes.shape[0] == 1
        assert gt_boxes.shape[2] == 5
        assert gt_boxes.dtype.kind == 'f'
        assert isinstance(gt_boxes, Variable)

        assert img_info.shape == (1, 2)
        assert img_info.dtype.kind == 'i'
        assert isinstance(img_info, Variable)

    def __call__(self, feat_h, feat_w, gt_boxes, img_info):
        """Calc targets of classification labels and bbox regression.

        Args:
            feat_h (int): The height of feature map.
            feat_w (int): The width of feature map.
            gt_boxes (:class:`~chainer.Variable`): The ground truth bounding
                boxes and its class label array. The shape should be
                :math:`(1, n_gt_boxes, 5)` and the batchsize should be 1.
                Each 5-dimensional vector has :math:`(x1, y1, x2, y2, cls_id)`.
                The scale of these values is at the input image scale.
            img_info (:class:`~chainer.Variable`): The input image info. It
                contains :math:`(height, width)` and the batchsize should be 1.
                So the shape should be :math:`(1, 2)`.
                
        Returns:
            bbox_labels (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                Classification labels of all anchor boxes. It contains values
                from :math:`{-1, 0, 1}` and the numbers of negative (=0) and
                positive (=1) are the same.
            bbox_reg_targets (:class:`~numpy.ndarray` or
                    :class:`~cupy.ndarray`):
                The regression targets of bounding box transformation
                parameters.
            inds_inside (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                Indices of all anchor boxes that inside of the input image out
                of all possible anchor boxes (`all_anchors`) that has
                :math:`K \times A` anchor boxes. This should be used to select
                proposals to be compared with the above two targets.
            n_all_bbox (int): The number of all possible bbox. This value
                is always larger than `len(inds_inside)`.

        """
        if self.type_check_enable:
            self._check_data_type_forward(gt_boxes, img_info)

        # Currently it assumes that the batchsize is always 1
        gt_boxes = gt_boxes.data[0]
        img_info = img_info.data[0]

        with cuda.get_device_from_array(gt_boxes):
            # (feat_h x feat_w x n_anchors, 4)
            xp = cuda.get_array_module(gt_boxes)
            all_bbox = xp.asarray(self._generate_all_bbox(feat_h, feat_w))
            inds_inside, all_inside_bbox = keep_inside(all_bbox, img_info)
            argmax_overlaps_inds, bbox_labels = \
                self._create_bbox_labels(
                    inds_inside, all_inside_bbox, gt_boxes)

            # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
            gt_boxes = gt_boxes[argmax_overlaps_inds]
            bbox_reg_targets = bbox_transform(all_inside_bbox, gt_boxes)
            bbox_reg_targets = bbox_reg_targets.astype(xp.float32)

            return bbox_labels, bbox_reg_targets, inds_inside, len(all_bbox)

    def _create_bbox_labels(self, inds_inside, anchors, gt_boxes):
        """Create bbox labels.

        label: 1 is positive, 0 is negative, -1 is dont care
        """
        xp = cuda.get_array_module(inds_inside)

        # assign ignore labels first
        labels = xp.ones((len(inds_inside),), dtype=np.int32) * -1

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
            # TODO(mitmul): Current cupy.random.choice doesn't support
            #               replace=False
            fg_inds = cuda.to_cpu(fg_inds)
            disable_inds = np.random.choice(
                fg_inds, size=int(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - xp.sum(labels == 1)
        bg_inds = xp.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            # TODO(mitmul): Current cupy.random.choice doesn't support
            #               replace=False
            bg_inds = cuda.to_cpu(bg_inds)
            disable_inds = np.random.choice(
                bg_inds, size=int(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        # TODO(mitmul): Remove this when cupy.random.choice with
        #                replace=False becomes available
        labels = xp.asarray(labels)

        return argmax_overlaps_inds, labels

    def _calc_overlaps(self, anchors, gt_boxes, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        xp = cuda.get_array_module(anchors)

        # TODO(mitmul): Use bbox_overlaps for GPU
        anchors = cuda.to_cpu(anchors)
        gt_boxes = cuda.to_cpu(gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        # TODO(mitmul): Remove this when bbox_overlaps for GPU comes
        overlaps = xp.asarray(overlaps)

        argmax_overlaps_inds = overlaps.argmax(axis=1)
        gt_argmax_overlaps_inds = overlaps.argmax(axis=0)

        max_overlaps = overlaps[
            xp.arange(len(inds_inside)), argmax_overlaps_inds]
        gt_max_overlaps = overlaps[
            gt_argmax_overlaps_inds, xp.arange(overlaps.shape[1])]
        gt_argmax_overlaps_inds = xp.where(overlaps == gt_max_overlaps)[0]

        return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds
