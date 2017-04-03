# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from models.bbox_transform import bbox_transform_inv
from models.bbox_transform import clip_boxes
from models.region_proposal_network import RegionProposalNetwork
from models.vgg16 import VGG16Prev


class FasterRCNN(chainer.Chain):
    def __init__(
            self, trunk_class=VGG16Prev, rpn_in_ch=512, rpn_mid_ch=512,
            feat_stride=16, anchor_ratios=[0.5, 1, 2],
            anchor_scales=[8, 16, 32], num_classes=21):
        w = chainer.initializers.Normal(0.01)
        super(FasterRCNN, self).__init__(
            trunk=trunk_class(),
            RPN=RegionProposalNetwork(rpn_in_ch, rpn_mid_ch, feat_stride, anchor_ratios, anchor_scales, num_classes),
            fc6=L.Linear(None, 4096, initialW=w),
            fc7=L.Linear(4096, 4096, initialW=w),
            cls_score=L.Linear(4096, num_classes, initialW=w),
            bbox_pred=L.Linear(4096, num_classes * 4, initialW=w),
        )
        self._rcnn_train = True
        self.spatial_scale = 1. / feat_stride

    @property
    def rcnn_train(self):
        return self._rcnn_train

    @rcnn_train.setter
    def rcnn_train(self, val):
        self._rcnn_train = val

    @property
    def rpn_train(self):
        return self.RPN.train

    @rpn_train.setter
    def rpn_train(self, val):
        self.RPN.train = val

    def __call__(self, x, img_info, gt_boxes=None):
        assert x.shape[0] == 1, 'Batchsize should be 1 but was {}'.format(x.shape[0])

        xp = self.trunk.xp
        feature_map = self.trunk(x)

        if self.rpn_train and gt_boxes is not None:
            rpn_cls_loss, rpn_loss_bbox = self.RPN(feature_map, img_info, gt_boxes)
            reporter.report({'rpn_cls_loss': rpn_cls_loss,
                             'rpn_loss_bbox': rpn_loss_bbox}, self)
            return rpn_cls_loss, rpn_loss_bbox
        else:
            rois, probs = self.RPN(feature_map, img_info, gt_boxes)

        # RCNN
        brois = xp.concatenate((xp.zeros((len(rois), 1), dtype=xp.float32), rois), axis=1)
        pool5 = F.roi_pooling_2d(feature_map, brois, 7, 7, self.spatial_scale)
        fc6 = F.dropout(F.relu(self.fc6(pool5)), train=self.rcnn_train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=self.rcnn_train)

        # Per class probability
        cls_score = self.cls_score(fc7)

        # BBox predictions
        bbox_pred = self.bbox_pred(fc7)
        bbox_deltas = bbox_pred.data

        if self.rcnn_train and gt_boxes is not None:
            loss_cls = F.softmax_cross_entropy(cls_score, gt_boxes[:, -1].astype(xp.int32))
            loss_bbox = F.huber_loss(bbox_deltas, gt_boxes[:, :4], 1)
            reporter.report({'loss_bbox': loss_bbox,
                             'loss_cls': loss_cls}, self)
            return loss_cls, loss_bbox
        else:
            pred_boxes = bbox_transform_inv(rois, bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, img_info)

            return F.softmax(cls_score), pred_boxes
