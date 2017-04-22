#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Mofidied by:
# Copyright (c) 2016 Shunta Saito

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

from chainer import cuda


def bbox_transform(ex_rois, gt_rois):
    xp = cuda.get_array_module(ex_rois)

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = xp.log(gt_widths / ex_widths)
    targets_dh = xp.log(gt_heights / ex_heights)

    targets = xp.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, trans):
    with cuda.get_device_from_array(boxes) as d:
        xp = cuda.get_array_module(boxes)
        if d.id >= 0:
            trans = cuda.to_gpu(trans, d)
            assert trans.device == boxes.device

        if boxes.shape[0] == 0:
            return xp.zeros((0, trans.shape[1]), dtype=trans.dtype)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = trans[:, 0::4]
        dy = trans[:, 1::4]
        dw = trans[:, 2::4]
        dh = trans[:, 3::4]

        pred_ctr_x = dx * widths[:, xp.newaxis] + ctr_x[:, xp.newaxis]
        pred_ctr_y = dy * heights[:, xp.newaxis] + ctr_y[:, xp.newaxis]
        pred_w = xp.exp(dw) * widths[:, xp.newaxis]
        pred_h = xp.exp(dh) * heights[:, xp.newaxis]

        pred_boxes = xp.zeros(trans.shape, dtype=trans.dtype)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


def clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    with cuda.get_device_from_array(boxes) as d:
        xp = cuda.get_array_module(boxes)
        if d.id >= 0:
            im_shape = cuda.to_gpu(im_shape, d)
            assert boxes.device == im_shape.device

        # x1 >= 0
        boxes[:, 0::4] = xp.maximum(
            xp.minimum(boxes[:, 0::4], int(im_shape[1] - 1)), 0)
        # y1 >= 0
        boxes[:, 1::4] = xp.maximum(
            xp.minimum(boxes[:, 1::4], int(im_shape[0] - 1)), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = xp.maximum(
            xp.minimum(boxes[:, 2::4], int(im_shape[1] - 1)), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = xp.maximum(
            xp.minimum(boxes[:, 3::4], int(im_shape[0] - 1)), 0)
        return boxes


def filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    with cuda.get_device_from_array(boxes):
        xp = cuda.get_array_module(boxes)
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = xp.where((ws >= min_size) & (hs >= min_size))[0]
        return keep


def keep_inside(anchors, img_info):
    """Calc indicies of anchors which are inside of the image size.

    Calc indicies of anchors which are located completely inside of the image
    whose size is speficied by img_info ((height, width, scale)-shaped array).
    """
    with cuda.get_device_from_array(anchors) as d:
        xp = cuda.get_array_module(anchors)
        if d.id >= 0:
            img_info = cuda.to_gpu(img_info, d)
            assert anchors.device == img_info.device

        inds_inside = xp.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] < img_info[1]) &  # width
            (anchors[:, 3] < img_info[0])  # height
        )[0]
        return inds_inside, anchors[inds_inside]
