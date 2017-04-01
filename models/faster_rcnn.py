# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import reporter
from chainer.cuda import to_gpu
from models.bbox_transform import bbox_transform_inv
from models.bbox_transform import clip_boxes
from models.region_proposal_network import RegionProposalNetwork
from models.vgg16 import VGG16
from models.vgg16 import VGG16Prev


class FasterRCNN(chainer.Chain):
    def __init__(
            self, trunk_class=VGG16Prev, rpn_in_ch=512, rpn_mid_ch=512,
            feat_stride=16, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
            num_classes=21):
        super(FasterRCNN, self).__init__(
            trunk=trunk_class(),
            RPN=RegionProposalNetwork(rpn_in_ch, rpn_mid_ch, feat_stride, anchor_ratios, anchor_scales, num_classes),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(4096, 4096),
            cls_score=L.Linear(4096, num_classes),
            bbox_pred=L.Linear(4096, num_classes * 4),
        )
        self._train = True
        self.spatial_scale = 1. / feat_stride

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val
        self.RPN.train = val

    def __call__(self, x, img_info, gt_boxes=None):
        assert x.shape[0] == 1, 'Batchsize should be 1 but was {}'.format(x.shape[0])

        xp = self.trunk.xp
        feature_map = self.trunk(x)

        if self.train and gt_boxes is not None:
            rpn_cls_loss, rpn_loss_bbox, rois = self.RPN(feature_map, img_info, gt_boxes)
        else:
            rois, probs = self.RPN(feature_map, img_info, gt_boxes)

        # RCNN
        brois = xp.concatenate((xp.zeros((len(rois), 1), dtype=xp.float32), rois), axis=1)
        pool5 = F.roi_pooling_2d(feature_map, brois, 7, 7, self.spatial_scale)
        fc6 = F.dropout(F.relu(self.fc6(pool5)), train=self.train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=self.train)

        # Per class probability
        cls_score = self.cls_score(fc7)

        # BBox predictions
        bbox_pred = self.bbox_pred(fc7)
        box_deltas = bbox_pred.data

        if self.train:
            if self.gpu >= 0:
                def tg(x): return to_gpu(x, device=self.gpu)
                labels = tg(labels)
                bbox_targets = tg(bbox_targets)
                bbox_inside_weights = tg(bbox_inside_weights)
                bbox_outside_weights = tg(bbox_outside_weights)
            loss_cls = F.softmax_cross_entropy(cls_score, labels)

            # huber_loss is smooth_l1_loss when delta = 1
            loss_bbox = F.huber_loss(bbox_pred, bbox_targets, 1.0)
            reporter.report({'rpn_loss_cls': rpn_cls_loss,
                             'rpn_loss_bbox': rpn_loss_bbox,
                             'loss_bbox': loss_bbox,
                             'loss_cls': loss_cls}, self)

            return rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls
        else:
            pred_boxes = bbox_transform_inv(rois, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, img_info)

            return F.softmax(cls_score), pred_boxes
