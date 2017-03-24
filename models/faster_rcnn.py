# Copyright (c) 2016 Shunta Saito

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import reporter
from chainer.cuda import to_gpu
from chainer.links import VGG16Layers
from lib.faster_rcnn.bbox_transform import bbox_transform_inv
from lib.faster_rcnn.bbox_transform import clip_boxes
from lib.faster_rcnn.proposal_target_layer import ProposalTargetLayer
from lib.faster_rcnn.smooth_l1_loss import smooth_l1_loss
from lib.models.rpn import RPN
from lib.models.vgg16 import VGG16


class FasterRCNN(chainer.Chain):

    def __init__(
            self, gpu=-1, trunk=VGG16, rpn_in_ch=512, rpn_out_ch=512,
            n_anchors=9, feat_stride=16, anchor_scales=[8, 16, 32],
            num_classes=21, spatial_scale=0.0625, rpn_sigma=1.0, sigma=3.0):
        super(FasterRCNN, self).__init__()
        trunk = trunk()
        if isinstance(trunk, VGG16Layers):
            trunk._children.remove('fc6')
            trunk._children.remove('fc7')
            trunk._children.remove('fc8')
            del trunk.fc6
            del trunk.fc7
            del trunk.fc8
            del trunk.functions['fc6']
            del trunk.functions['fc7']
            del trunk.functions['fc8']
            del trunk.functions['prob']
            self.trunk_key = 'pool5'
        self.add_link('trunk', trunk)
        self.add_link('RPN', RPN(rpn_in_ch, rpn_out_ch, n_anchors, feat_stride,
                                 anchor_scales, num_classes, rpn_sigma))
        self.add_link('fc6', L.Linear(25088, 4096))
        self.add_link('fc7', L.Linear(4096, 4096))
        self.add_link('cls_score', L.Linear(4096, num_classes))
        self.add_link('bbox_pred', L.Linear(4096, num_classes * 4))
        self.train = True
        self.gpu = gpu
        self.sigma = sigma

        self.spatial_scale = spatial_scale
        self.proposal_target_layer = ProposalTargetLayer(num_classes)

    def __call__(self, x, im_info, gt_boxes=None):
        if hasattr(self, 'trunk_key'):
            h = self.trunk(x, [self.trunk_key])[self.trunk_key]
        else:
            h = self.trunk(x)
        if isinstance(im_info, chainer.Variable):
            im_info = chainer.cuda.to_cpu(im_info.data)
        if self.train and gt_boxes is not None:
            if isinstance(gt_boxes, chainer.Variable):
                gt_boxes = chainer.cuda.to_cpu(gt_boxes.data)
            rpn_cls_loss, rpn_loss_bbox, rois = self.RPN(
                h, im_info, self.gpu, gt_boxes)
        else:
            rois = self.RPN(h, im_info, self.gpu, gt_boxes)

        if self.train:
            rois, labels, bbox_targets, bbox_inside_weights, \
                bbox_outside_weights = self.proposal_target_layer(
                    rois, gt_boxes)

        # Convert rois
        if self.gpu >= 0:
            rois = to_gpu(rois, device=self.gpu)
            im_info = to_gpu(im_info, device=self.gpu)
            with chainer.cuda.Device(self.gpu):
                boxes = rois[:, 1:5] / im_info[0][2]
        else:
            boxes = rois[:, 1:5] / im_info[0][2]

        # RCNN
        pool5 = F.roi_pooling_2d(
            self.trunk.feature, rois, 7, 7, self.spatial_scale)
        fc6 = F.dropout(F.relu(self.fc6(pool5)), train=self.train)
        fc7 = F.dropout(F.relu(self.fc7(fc6)), train=self.train)

        # Per class probability
        cls_score = self.cls_score(fc7)
        cls_prob = F.softmax(cls_score)

        # BBox predictions
        bbox_pred = self.bbox_pred(fc7)
        box_deltas = bbox_pred.data

        if self.train:
            if self.gpu >= 0:
                tg = lambda x: to_gpu(x, device=self.gpu)
                labels = tg(labels)
                bbox_targets = tg(bbox_targets)
                bbox_inside_weights = tg(bbox_inside_weights)
                bbox_outside_weights = tg(bbox_outside_weights)
            loss_cls = F.softmax_cross_entropy(cls_score, labels)
            loss_bbox = smooth_l1_loss(
                bbox_pred, bbox_targets, bbox_inside_weights,
                bbox_outside_weights, self.sigma)

            reporter.report({'rpn_loss_cls': rpn_cls_loss,
                             'rpn_loss_bbox': rpn_loss_bbox,
                             'loss_bbox': loss_bbox,
                             'loss_cls': loss_cls}, self)

            return rpn_cls_loss, rpn_loss_bbox, loss_bbox, loss_cls
        else:
            pred_boxes = bbox_transform_inv(boxes, box_deltas, self.gpu)
            pred_boxes = clip_boxes(pred_boxes, im_info[0][:2], self.gpu)

            return cls_prob, pred_boxes
