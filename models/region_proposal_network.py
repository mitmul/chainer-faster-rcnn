#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.cuda import to_gpu
from models.anchor_target_layer import AnchorTargetLayer
from models.proposal_layer import ProposalLayer
from models.smooth_l1_loss import smooth_l1_loss


class RegionProposalNetwork(chainer.Chain):

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
        rpn_sigma (float): Used in calculation of smooth_l1_loss.
        train (bool): If True, anchor target layer will be instantiated.

    """

    def __init__(
            self, in_ch=512, mid_ch=512, feat_stride=16,
            anchor_ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32),
            num_classes=21):
        w = chainer.initializers.Normal(0.01)
        n_anchors = len(anchor_ratios) * len(anchor_scales)
        super(RegionProposalNetwork, self).__init__(
            rpn_conv_3x3=L.Convolution2D(in_ch, mid_ch, 3, 1, 1, initialW=w),
            rpn_cls_score=L.Convolution2D(mid_ch, 2 * n_anchors, 1, 1, 0, initialW=w),
            rpn_bbox_pred=L.Convolution2D(mid_ch, 4 * n_anchors, 1, 1, 0, initialW=w)
        )
        self.proposal_layer = ProposalLayer(feat_stride, anchor_ratios, anchor_scales)
        self.anchor_target_layer = ProposalLayer(feat_stride, anchor_ratios, anchor_scales)
        self._train = True

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val
        self.proposal_layer.train = val

    def __call__(self, x, img_info, gt_boxes=None):
        """Calculate RoIs or losses and RoIs.

        Args:
            x (:class:~`Variable` or :class:`~numpy.ndarray` or \
                    :class:`~cupy.ndarray`): Input feature maps.
            img_info (list of integers): The input image size in
                :math:`(img_h, img_w)`.
            gt_boxes (:class:`~numpy.ndarray` or :class:`~cupy.ndarray`):
                :math:`(n_boxes, x1, y1, x2, y2)`-shaped array. The scale is
                at the input image scale.

        """
        assert x.shape[0] == 1, 'Batchsize should be 1 but was {}'.format(x.shape[0])

        h = F.relu(self.rpn_conv_3x3(x))
        rpn_cls_score = self.rpn_cls_score(h)  # 1 * 2 * n_anchors x feat_h x feat_w
        rpn_cls_prob = F.softmax(rpn_cls_score)
        rpn_bbox_pred = self.rpn_bbox_pred(h)  # 1 * 4 * n_anchors x feat_h x feat_w

        # Predicted RoI proposals
        rois, probs = self.proposal_layer(rpn_cls_prob.data[0], rpn_bbox_pred.data[0], img_info)

        if self.train and gt_boxes is not None:
            bbox_labels, bbox_reg_targets = self.anchor_target_layer(gt_boxes, img_info)

            rpn_cls_loss = F.softmax_cross_entropy(rpn_cls_score, bbox_labels)
            rpn_loss_bbox = F.huber_loss(rpn_bbox_pred, bbox_reg_targets, 1)
            return rpn_cls_loss, rpn_loss_bbox, rois
        else:
            return rois, probs
