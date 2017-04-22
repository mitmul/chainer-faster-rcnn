#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import Variable
from chainer import cuda
from chainer import initializers
from chainer import reporter
from models.anchor_target_layer import AnchorTargetLayer
from models.proposal_layer import ProposalLayer


class RegionProposalNetwork(Chain):
    """Region Proposal Network

    It generates predicted class probabilities and bounding box regression
    params from the shared convnet (trunk model).

    It performs two different convolutional layers that intend to calculate
    class probabilities for each pixel on the input feature map and bounding
    box regression parameters for each pixel on the input feature map,
    respectively.

    Args:
        mid_ch (int): The output channel of the first convolution layer before
            the two convs for class probability and bbox regression params. It
            means the dimension of a shared feature vector for following
            cls/reg networks.
        feat_stride (int): The stride of the output of the first conv layer
            in the input image plane, e.g., 16 for conv5_3 of VGG16.
        anchor_scales (list of integers): The scales of anchor in pixel size.
            The size is in the scale of the input image plane for the trunk
            model.
        num_classes (int): The output dimension of class probability.
        lambda (float): A balancing parameter betwee `rpn_cls_loss` and
            `rpn_loss_bbox`

    """

    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __init__(
            self, in_ch=512, mid_ch=512, feat_stride=16,
            anchor_ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32),
            num_classes=21, loss_lambda=1., delta=3):
        w = initializers.Normal(0.01)
        n_anchors = len(anchor_ratios) * len(anchor_scales)
        super(RegionProposalNetwork, self).__init__(
            rpn_conv_3x3=L.Convolution2D(in_ch, mid_ch, 3, 1, 1, initialW=w),
            rpn_cls_score=L.Convolution2D(
                mid_ch, 2 * n_anchors, 1, 1, 0, initialW=w),
            rpn_bbox_pred=L.Convolution2D(
                mid_ch, 4 * n_anchors, 1, 1, 0, initialW=w)
        )
        self.proposal_layer = ProposalLayer(
            feat_stride, anchor_ratios, anchor_scales)
        self.anchor_target_layer = AnchorTargetLayer(
            feat_stride, anchor_ratios, anchor_scales)
        self._loss_lambda = loss_lambda
        self._train = True
        self._delta = delta

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val
        self.proposal_layer.train = val

    def _check_data_type_forward(self, x, img_info, gt_boxes):
        assert x.shape[0] == 1
        assert x.dtype.kind == 'f'
        assert img_info.shape == (1, 2)
        assert img_info.dtype.kind == 'i'
        assert isinstance(x, Variable)
        assert isinstance(img_info, Variable)
        if gt_boxes is not None:
            assert gt_boxes.shape[0] == 1
            assert gt_boxes.shape[2] == 5
            assert gt_boxes.dtype.kind == 'f'
            assert isinstance(gt_boxes, Variable)

    def __call__(self, x, img_info, gt_boxes=None):
        """Calculate RoIs or losses and RoIs.

        Args:
            x (:class:`~chainer.Variable`): Input feature maps. The shape
                should be :math:`(1, n_feat_channels, feat_h, feat_w)`.
            img_info (:class:`~chainer.Variable`): The input image size
                represented as a list of integers such as
                :math:`(img_h, img_w)`. And the batchsize should be 1, so the
                shape should be :math:`(1, 2)`.
            gt_boxes (:class:`~chainer.Variable` or None):
                :math:`(1, n_gt_boxes, x1, y1, x2, y2)`-shaped 6-dimensional
                array. The scale is at the input image scale. Default value is
                `None`.

        Returns:
            If self.train and gt_boxes is not None:
                rpn_cls_loss (:class:`~chainer.Variable`)
                rpn_loss_bbox (:class:`~chainer.Variable`)
            elif gt_boxes is not None:
                proposals (:class:`~numpy.ndaarray` or :class:`~cupy.ndarray`)
                probs (:class:`~numpy.ndaarray` or :class:`~cupy.ndarray`)

        """
        if self.type_check_enable:
            self._check_data_type_forward(x, img_info, gt_boxes)

        # Network fowarding
        h = F.relu(self.rpn_conv_3x3(x))
        rpn_cls_score = self.rpn_cls_score(h)
        rpn_cls_prob = F.softmax(rpn_cls_score)
        rpn_bbox_pred = self.rpn_bbox_pred(h)

        # Predicted RoI proposals
        proposals, probs = self.proposal_layer(
            rpn_cls_prob, rpn_bbox_pred, img_info)

        if self.train and gt_boxes is not None:
            # Get feature map size
            feat_h, feat_w = rpn_cls_prob.shape[2:]

            # Get target values to calc losses
            bbox_labels, bbox_reg_targets, inds_inside, n_all_bbox = \
                self.anchor_target_layer(feat_h, feat_w, gt_boxes, img_info)

            # Calc classification loss
            rpn_loss_cls, rpn_cls_accuracy = self._calc_rpn_loss_cls(
                rpn_cls_score, bbox_labels, inds_inside, n_all_bbox, feat_h,
                feat_w)
            rpn_loss_cls.name = 'rpn_loss_cls'

            # Calc regression loss
            rpn_loss_bbox = self._calc_rpn_loss_bbox(
                rpn_bbox_pred, bbox_reg_targets, inds_inside)
            rpn_loss_bbox.name = 'rpn_loss_bbox'

            rpn_loss = rpn_loss_cls + self._loss_lambda * rpn_loss_bbox
            rpn_loss.name = 'rpn_loss'

            try:
                reporter.report({'rpn_loss_cls': rpn_loss_cls,
                                 'rpn_cls_accuracy': rpn_cls_accuracy,
                                 'rpn_loss_bbox': rpn_loss_bbox,
                                 'rpn_loss': rpn_loss}, self)
            except:
                pass

            return rpn_loss

        return proposals, probs

    def _calc_rpn_loss_cls(self, rpn_cls_score, bbox_labels, inds_inside,
                           n_all_bbox, feat_h, feat_w):
        # To map up
        with cuda.get_device_from_array(bbox_labels):
            xp = cuda.get_array_module(bbox_labels)
            bbox_labels_mapped = xp.ones((n_all_bbox,), dtype=xp.int32) * -1
            bbox_labels_mapped[inds_inside] = bbox_labels

            # Initially it's in (K x A, 4), so the number of lattices first
            n_anchors = self.proposal_layer._num_anchors
            bbox_labels_mapped = bbox_labels_mapped.reshape(1, feat_h, feat_w,
                                                            n_anchors)
            bbox_labels_mapped = bbox_labels_mapped.transpose(0, 3, 1, 2)

            # Classification loss (bg/fg)
            rpn_cls_score = rpn_cls_score.reshape(1, 2, n_anchors, feat_h,
                                                  feat_w)
            rpn_loss_cls = F.softmax_cross_entropy(
                rpn_cls_score, bbox_labels_mapped)
            rpn_cls_accuracy = F.accuracy(
                rpn_cls_score, bbox_labels_mapped, -1)
            return rpn_loss_cls.reshape(()), rpn_cls_accuracy.reshape(())

    def _calc_rpn_loss_bbox(self, rpn_bbox_pred, bbox_reg_targets, inds_inside):
        # rpn_bbox_pred has the shape of (1, 4 x n_anchors, feat_h, feat_w)
        n_anchors = self.proposal_layer._num_anchors
        # Reshape it into (4, A, K)
        rpn_bbox_pred = rpn_bbox_pred.reshape(4, n_anchors, -1)
        # Transpose it into (K, A, 4)
        rpn_bbox_pred = rpn_bbox_pred.transpose(2, 1, 0)
        # Reshape it into (K x A, 4)
        rpn_bbox_pred = rpn_bbox_pred.reshape(-1, 4)
        # Keep the number of bbox
        n_bbox = rpn_bbox_pred.shape[0]
        # Select bbox and ravel it
        rpn_bbox_pred = F.flatten(rpn_bbox_pred[inds_inside])
        # Create batch dimension
        rpn_bbox_pred = F.expand_dims(rpn_bbox_pred, 0)
        # Ravel the targets and create batch dimension
        bbox_reg_targets = bbox_reg_targets.ravel()[None, :]
        # Calc Smooth L1 Loss (When delta=1, huber loss is SmoothL1Loss)
        rpn_loss_bbox = F.huber_loss(rpn_bbox_pred, bbox_reg_targets,
                                     self._delta)
        rpn_loss_bbox /= n_bbox
        return rpn_loss_bbox.reshape(())
