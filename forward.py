#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import serializers
from chainer.cuda import to_gpu
from lib.cpu_nms import cpu_nms as nms
from lib.models.faster_rcnn import FasterRCNN

import argparse
import chainer
import cv2 as cv
import numpy as np

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


def get_model(gpu):
    model = FasterRCNN(gpu)
    model.train = False
    serializers.load_npz('data/VGG16_faster_rcnn_final.model', model)

    return model


def img_preprocessing(orig_img, pixel_means, max_size=1000, scale=600):
    img = orig_img.astype(np.float32, copy=True)
    img -= pixel_means
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    im_scale = float(scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv.INTER_LINEAR)

    return img.transpose([2, 0, 1]).astype(np.float32), im_scale


def draw_result(out, im_scale, clss, bbox, nms_thresh, conf):
    CV_AA = 16
    for cls_id in range(1, 21):
        _cls = clss[:, cls_id][:, np.newaxis]
        _bbx = bbox[:, cls_id * 4: (cls_id + 1) * 4]
        dets = np.hstack((_bbx, _cls))
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= conf)[0]
        for i in inds:
            x1, y1, x2, y2 = map(int, dets[i, :4])
            cv.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, CV_AA)
            ret, baseline = cv.getTextSize(
                CLASSES[cls_id], cv.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            cv.rectangle(out, (x1, y2 - ret[1] - baseline),
                         (x1 + ret[0], y2), (0, 0, 255), -1)
            cv.putText(out, CLASSES[cls_id], (x1, y2 - baseline),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, CV_AA)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_fn', type=str, default='sample.jpg')
    parser.add_argument('--out_fn', type=str, default='result.jpg')
    parser.add_argument('--nms_thresh', type=float, default=0.3)
    parser.add_argument('--conf', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    xp = chainer.cuda.cupy if chainer.cuda.available and args.gpu >= 0 else np
    model = get_model(gpu=args.gpu)
    if chainer.cuda.available and args.gpu >= 0:
        model.to_gpu(args.gpu)

    orig_image = cv.imread(args.img_fn)
    img, im_scale = img_preprocessing(orig_image, PIXEL_MEANS)
    img = np.expand_dims(img, axis=0)
    if args.gpu >= 0:
        img = to_gpu(img, device=args.gpu)
    img = chainer.Variable(img, volatile=True)
    h, w = img.data.shape[2:]
    cls_score, bbox_pred = model(img, np.array([[h, w, im_scale]]))
    cls_score = cls_score.data

    if args.gpu >= 0:
        cls_score = chainer.cuda.cupy.asnumpy(cls_score)
        bbox_pred = chainer.cuda.cupy.asnumpy(bbox_pred)
    result = draw_result(orig_image, im_scale, cls_score, bbox_pred,
                         args.nms_thresh, args.conf)
    cv.imwrite(args.out_fn, result)
