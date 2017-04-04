#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import cv2 as cv

import matplotlib  # isort:skip
matplotlib.use('Agg')
import chainercv  # isort:skip


class VOC(chainercv.datasets.VOCDetectionDataset):

    LABELS = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
    IMG_TARGET_SIZE = 600
    IMG_MAX_SIZE = 1000

    def __init__(self, mode='train', use_difficult=False):
        super(VOC, self).__init__(mode=mode, use_difficult=use_difficult)
        self.mean = np.array([[[103.939, 116.779, 123.68]]])  # BGR

    def get_example(self, i):
        img, bbox, label = super(VOC, self).get_example(i)
        img = img.transpose(1, 2, 0)
        img -= self.mean

        # Scaling
        im_size_min = np.min(img.shape[:2])
        im_size_max = np.max(img.shape[:2])
        im_scale = float(self.IMG_TARGET_SIZE) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.IMG_MAX_SIZE:
            im_scale = float(self.IMG_MAX_SIZE) / float(im_size_max)
        img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv.INTER_CUBIC)
        img = img.transpose(2, 0, 1).astype(np.float32)
        bbox *= im_scale
        bbox = np.concatenate(
            (bbox, label[:, None]), axis=1).astype(np.float32)
        return img, np.asarray(img.shape[1:]), bbox


if __name__ == '__main__':
    dataset = VOC('train')
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))

    dataset = VOC('val')
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))
