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
# --------------------------------------------------------

import numpy as np
import six


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=[4, 8, 16, 32], xp=np):
    # Generate anchor (reference) windows by enumerating aspect ratios X
    # scales wrt a reference (0, 0, 15, 15) window.
    base_anchor = xp.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, xp.asarray(ratios), xp)
    anchors = xp.vstack([_scale_enum(ratio_anchors[i, :], xp.asarray(scales), xp)
                         for i in six.moves.range(len(ratio_anchors))])
    return anchors


def _whctrs(anchor):
    # Return width, height, x center, and y center for an anchor (window).
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr, xp=np):
    # Given a vector of widths (ws) and heights (hs) around a center
    # (x_ctr, y_ctr), output a set of anchors (windows).
    ws, hs = ws[:, None], hs[:, None]
    anchors = xp.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios, xp=np):
    # Enumerate a set of anchors for each aspect ratio wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = xp.rint(xp.sqrt(size_ratios))
    hs = xp.rint(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr, xp)
    return anchors


def _scale_enum(anchor, scales, xp=np):
    # Enumerate a set of anchors for each scale wrt an anchor.
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr, xp)
    return anchors
