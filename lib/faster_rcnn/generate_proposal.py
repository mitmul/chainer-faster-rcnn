# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# https://github.com/rbgirshick/py-faster-rcnn
# --------------------------------------------------------

import chainer
import numpy as np
from generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms


class ProposalLayer(object):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    TEST_RPN_PRE_NMS_TOP_N = -1
    TEST_RPN_POST_NMS_TOP_N = 2000
    TEST_RPN_NMS_THRESH = 0.7
    TEST_RPN_MIN_SIZE = 16

    TRAIN_RPN_PRE_NMS_TOP_N = 12000
    TRAIN_RPN_POST_NMS_TOP_N = 2000
    TRAIN_RPN_NMS_THRESH = 0.7
    TRAIN_RPN_MIN_SIZE = 16

    def __init__(self, feat_stride=16):
        self.feat_stride = 16
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.shape[0]
        self.train = False

    def __call__(self, x, bbox_deltas, im_info):
        if isinstance(bbox_deltas.data, chainer.cuda.ndarray):
            bbox_deltas = chainer.cuda.to_cpu(bbox_deltas.data)
        if isinstance(x.data, chainer.cuda.ndarray):
            x = chainer.cuda.to_cpu(x.data)

        assert x.shape[0] == 1, 'Only single item batches are supported'

        if self.train:
            pre_nms_topN = self.TRAIN_RPN_PRE_NMS_TOP_N
            post_nms_topN = self.TRAIN_RPN_POST_NMS_TOP_N
            nms_thresh = self.TRAIN_RPN_NMS_THRESH
            min_size = self.TRAIN_RPN_MIN_SIZE
        else:
            pre_nms_topN = self.TEST_RPN_PRE_NMS_TOP_N
            post_nms_topN = self.TEST_RPN_POST_NMS_TOP_N
            nms_thresh = self.TEST_RPN_NMS_THRESH
            min_size = self.TEST_RPN_MIN_SIZE

        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs, which we want
        scores = x[:, self.num_anchors:, :, :]

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self.feat_stride
        shift_y = np.arange(0, height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self.num_anchors
        K = shifts.shape[0]
        anchors = self.anchors.reshape((1, A, 4)) + \
            shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack(
            (batch_inds, proposals.astype(np.float32, copy=False)))
        blob = chainer.cuda.cupy.asarray(blob, np.float32)
        rois = chainer.Variable(blob, volatile=not self.train)

        return rois


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
