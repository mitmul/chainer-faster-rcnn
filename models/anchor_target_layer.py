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

import numpy

import chainer
from models.proposal_layer import ProposalLayer
from models.bbox_transform import clip_boxes
from models.bbox import bbox_overlaps


class AnchorTargetLayer(ProposalLayer):
    """Assign anchors to ground-truth targets

    Produces 1) anchor classification labels and 2) bounding-box regression
    targets.

    Args:
        feat_stride (int): The stride of corresponding pixels in the lowest
            layer (image). A couple of adjacent values on the input feature map
            are actually distant from each other with `feat_stride` pixels in
            the image plane.
        scales (list): A list of scales of anchor boxes. See
    """

    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_CLOBBER_POSITIVES = False
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 256
    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    RPN_POSITIVE_WEIGHT = -1.0

    def __init__(self, feat_stride=16, anchor_scales=[4, 8, 16, 32],
                 allowed_border=0, xp=numpy):
        super(AnchorTargetLayer, self).__init__(feat_stride, anchor_scales, xp)
        self.allowed_border = allowed_border

    def calc_overlaps(self, anchors, gt_boxes, inds_inside):
        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)

        # TODO(mitmul): Implement GPU version of bbox_overlaps
        anchors = chainer.cuda.to_cpu(anchors)
        gt_boxes = chainer.cuda.to_cpu(gt_boxes)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        anchors = self.xp.asarray(anchors)
        gt_boxes = self.xp.asarray(gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[
            self.xp.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   self.xp.arange(overlaps.shape[1])]
        gt_argmax_overlaps = self.xp.where(overlaps == gt_max_overlaps)[0]

        return (argmax_overlaps, max_overlaps,
                gt_max_overlaps, gt_argmax_overlaps)

    def create_labels(self, inds_inside, anchors, gt_boxes):
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = self.xp.empty((len(inds_inside),), dtype=self.xp.float32)
        labels.fill(-1)

        _ = self.calc_overlaps(anchors, gt_boxes, inds_inside)
        argmax_overlaps, max_overlaps, gt_max_overlaps, gt_argmax_overlaps = _

        if not self.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.RPN_POSITIVE_OVERLAP] = 1

        if self.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber
            # positives
            labels[max_overlaps < self.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.RPN_FG_FRACTION * self.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        return argmax_overlaps, labels

    def calc_inside_weights(self, inds_inside, labels):
        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(
            self.RPN_BBOX_INSIDE_WEIGHTS)

        return bbox_inside_weights

    def calc_outside_weights(self, inds_inside, labels):
        bbox_outside_weights = np.zeros(
            (len(inds_inside), 4), dtype=np.float32)
        if self.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.RPN_POSITIVE_WEIGHT > 0) &
                    (self.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (self.RPN_POSITIVE_WEIGHT / np.sum(labels == 1))
            negative_weights = ((1.0 - self.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        return bbox_outside_weights

    def mapup_to_anchors(
            self, labels, total_anchors, inds_inside, bbox_targets,
            bbox_inside_weights, bbox_outside_weights):
        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(
            bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(
            bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(
            bbox_outside_weights, total_anchors, inds_inside, fill=0)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def __call__(self, x, gt_boxes, img_info):
        """It takes numpy or cupy arrays

        Args:
            x (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                A :math:`(ch, feat_h, feat_w)`-shaped feature map.
            gt_boxes (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(n_boxes, x1, y1, x2, y2)`-shaped array.
            img_info (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(3,)`-shaped list that contains
                :math:`(img_h, img_w, img_scale)`.
        """

        _, feat_h feat_w = x.shape
        anchors = self._generate_anchors(feat_h, feat_w)
        inds_inside, anchors = _keep_inside(anchors, img_info, self.allowed_border)

        argmax_overlaps, labels = self.create_labels(inds_inside, anchors, gt_boxes)

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = self._compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = self.calc_inside_weights(inds_inside, labels)
        bbox_outside_weights = self.calc_outside_weights(inds_inside, labels)

        # map up to original set of anchors
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            self.mapup_to_anchors(
                labels, total_anchors, inds_inside, bbox_targets,
                bbox_inside_weights, bbox_outside_weights)

        # labels
        labels = labels.reshape(
            (1, height, width, self.n_anchors)).transpose(0, 3, 1, 2)
        labels = labels.astype(np.int32)

        # bbox_targets
        bbox_targets = bbox_targets.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape(
            (1, height, width, self.n_anchors * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size count)"""

    if len(data.shape) == 1:
        ret = numpy.empty((count, ), dtype=numpy.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = numpy.empty((count, ) + data.shape[1:], dtype=numpy.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(
        numpy.float32, copy=False)


def _keep_inside(anchors, img_info, allowed_border=0):
    """Calc indicies of anchors which are inside of the image size.

    Calc indicies of anchors which are located completely inside of the image
    whose size is speficied by img_info ((height, width, scale)-shaped array).
    """

    inds_inside = np.where(
        (anchors[:, 0] >= -self.allowed_border) &
        (anchors[:, 1] >= -self.allowed_border) &
        (anchors[:, 2] < im_info[1] + self.allowed_border) &  # width
        (anchors[:, 3] < im_info[0] + self.allowed_border)    # height
    )[0]
    return inds_inside, anchors[inds_inside]
