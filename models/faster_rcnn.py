# Copyright (c) 2016 Shunta Saito

import os

import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import Variable
from chainer import cuda
from chainer import initializers
from chainer import reporter
from models.bbox_transform import bbox_transform_inv
from models.bbox_transform import clip_boxes
from models.proposal_target_layer import ProposalTargetLayer
from models.region_proposal_network import RegionProposalNetwork
from models.vgg16 import VGG16


class FasterRCNN(Chain):
    type_check_enable = int(os.environ.get('CHAINER_TYPE_CHECK', '1')) != 0

    def __init__(
            self, trunk_class=VGG16, rpn_in_ch=512, rpn_mid_ch=512,
            feat_stride=16, anchor_ratios=(0.5, 1, 2),
            anchor_scales=(8, 16, 32), num_classes=21, loss_lambda=1,
            rpn_delta=3, rcnn_delta=1):
        w = initializers.Normal(0.01)
        super(FasterRCNN, self).__init__(
            trunk=trunk_class(),
            RPN=RegionProposalNetwork(
                rpn_in_ch, rpn_mid_ch, feat_stride, anchor_ratios,
                anchor_scales, num_classes, loss_lambda, rpn_delta),
            fc6=L.Linear(None, 4096, initialW=w),
            fc7=L.Linear(4096, 4096, initialW=w),
            cls_score=L.Linear(4096, num_classes, initialW=w),
            bbox_pred=L.Linear(4096, num_classes * 4, initialW=w),
        )
        self._feat_stride = feat_stride
        self._anchor_ratios = anchor_ratios
        self._anchor_scales = anchor_scales
        self._num_classes = num_classes
        self.RPN.train = False
        self._rcnn_train = False
        self._spatial_scale = 1. / feat_stride
        self._rpn_delta = rpn_delta
        self._rcnn_delta = rcnn_delta

    @property
    def rcnn_train(self):
        return self._rcnn_train

    @rcnn_train.setter
    def rcnn_train(self, val):
        self._rcnn_train = val
        if val:
            self.RPN.train = not val
        if self.rcnn_train or self.rpn_train:
            self.trunk.train = True
        else:
            self.trunk.train = False

    @property
    def rpn_train(self):
        return self.RPN.train

    @rpn_train.setter
    def rpn_train(self, val):
        self.RPN.train = val
        if val:
            self._rcnn_train = not val
        if self.rcnn_train or self.rpn_train:
            self.trunk.train = True
        else:
            self.trunk.train = False

    def _check_data_type_forward(self, x, img_info, gt_boxes):
        assert x.shape[0] == 1
        assert x.dtype.kind == 'f'
        assert isinstance(x, Variable)

        assert img_info.shape == (1, 2)
        assert img_info.dtype.kind == 'i'
        assert isinstance(img_info, Variable)

        if gt_boxes is not None:
            assert gt_boxes.shape[0] == 1
            assert gt_boxes.shape[1] > 0
            assert gt_boxes.shape[2] == 5
            assert gt_boxes.dtype.kind == 'f'
            assert isinstance(gt_boxes, Variable)

    def __call__(self, x, img_info, gt_boxes=None):
        """Faster RCNN forward

        Args:
            x (:class:`~chainer.Variable`): The input image. Note that the
                batchsize should be 1. So the shape should be
                :math:`(1, n_channels, height, width)`.
            img_info (:class:`~chainer.Variable`): The input image info. It
                contains :math:`(height, width)` and the batchsize should be 1.
                So the shape should be :math:`(1, 2)`.
            gt_boxes (:class:`~chainer.Variable`): The ground truth bounding
                boxes and its class label array. The shape should be
                :math:`(1, n_gt_boxes, 5)` and the batchsize should be 1.

        """
        if self.type_check_enable:
            self._check_data_type_forward(x, img_info, gt_boxes)

        # Use the array module of the backend of trunk model
        with cuda.get_device_from_array(x.data):
            xp, feature_map = self.trunk.xp, self.trunk(x)

            # RPN training mode
            if self.rpn_train and gt_boxes is not None:
                return self.RPN(feature_map, img_info, gt_boxes)
            else:
                proposals, probs = self.RPN(feature_map, img_info, gt_boxes)
                self.rpn_proposals = proposals
                self.rpn_probs = probs

            # RCNN
            batch_id = xp.zeros((len(proposals), 1), dtype=xp.float32)
            brois = xp.concatenate((batch_id, proposals), axis=1)
            pool5 = F.roi_pooling_2d(feature_map, brois, 7, 7,
                                     self._spatial_scale)
            fc6 = F.dropout(F.relu(self.fc6(pool5)), train=self.rcnn_train)
            fc7 = F.dropout(F.relu(self.fc7(fc6)), train=self.rcnn_train)

            # Per class probability
            cls_score = self.cls_score(fc7)

            # BBox predictions
            bbox_pred = self.bbox_pred(fc7)

            if self.rcnn_train and gt_boxes is not None:
                # Create proposal target layer if not exsist
                if not hasattr(self, 'proposal_target_layer'):
                    self.proposal_target_layer = ProposalTargetLayer(
                        self._feat_stride, self._anchor_ratios,
                        self._anchor_scales, self._num_classes)
                use_gt_boxes, bbox_reg_targets, keep_inds = \
                    self.proposal_target_layer(proposals, gt_boxes)

                # TODO(mitmul): Remove this re-sending below vars to GPU
                xp = self.RPN.xp
                if xp is cuda.cupy:
                    use_gt_boxes = xp.asarray(use_gt_boxes)
                    bbox_reg_targets = xp.asarray(bbox_reg_targets)
                    keep_inds = xp.asarray(keep_inds)

                # Select predicted scores and calc loss
                cls_score = cls_score[keep_inds]
                cls_labels = use_gt_boxes[:, -1].astype(xp.int32)
                loss_cls = F.softmax_cross_entropy(cls_score, cls_labels)
                loss_cls = loss_cls.reshape(())
                cls_acc = F.accuracy(cls_score, cls_labels, -1)

                # Select predicted bbox transformations and calc loss
                bbox_pred = bbox_pred[keep_inds]
                loss_bbox = F.huber_loss(bbox_pred, bbox_reg_targets,
                                         self._rcnn_delta)
                loss_bbox = F.sum(loss_bbox) / loss_bbox.size
                loss_bbox = loss_bbox.reshape(())

                loss_rcnn = loss_cls + loss_bbox

                reporter.report({'loss_cls': loss_cls,
                                 'cls_accuracy': cls_acc,
                                 'loss_bbox': loss_bbox,
                                 'loss_rcnn': loss_rcnn}, self)

                return loss_rcnn

            pred_boxes = bbox_transform_inv(proposals, bbox_pred.data)
            pred_boxes = clip_boxes(pred_boxes, img_info.data[0])

            return F.softmax(cls_score), pred_boxes
