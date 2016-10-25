#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import cv2 as cv
import glob
import numpy as np
import os
import xml.etree.ElementTree as ET


class VOC(chainer.dataset.DatasetMixin):

    LABELS = ('__background__',  # always index 0
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor')
    IMG_TARGET_SIZE = 600
    IMG_MAX_SIZE = 1000

    def __init__(
            self, img_dir, anno_dir, list_dir, list_suffix, use_diff=False):
        self.mean = np.array([[[103.939, 116.779, 123.68]]])  # BGR
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.use_diff = use_diff
        self.use_list = []
        for fn in glob.glob('{}/*_{}.txt'.format(list_dir, list_suffix)):
            for line in open(fn):
                self.use_list.append(line.strip().split()[0])
        self.parse_anno()

    def parse_anno(self):
        self.objects = []
        for fn in glob.glob('{}/*.xml'.format(self.anno_dir)):
            tree = ET.parse(fn)
            filename = tree.find('filename').text
            img_id = os.path.splitext(filename)[0]
            if img_id not in self.use_list:
                continue
            for obj in tree.findall('object'):
                if not self.use_diff and int(obj.find('difficult').text) == 1:
                    continue
                bb = obj.find('bndbox')
                bbox = [int(bb.find('xmin').text), int(bb.find('ymin').text),
                        int(bb.find('xmax').text), int(bb.find('ymax').text)]
                bbox = [float(b - 1) for b in bbox]
                datum = {
                    'filename': filename,
                    'name': obj.find('name').text.lower().strip(),
                    'pose': obj.find('pose').text.lower().strip(),
                    'truncated': int(obj.find('truncated').text),
                    'difficult': int(obj.find('difficult').text),
                    'bndbox': bbox,
                }
                self.objects.append(datum)

    def __len__(self):
        return len(self.objects)

    def get_example(self, i):
        obj = self.objects[i]
        bbox = obj['bndbox']
        name = obj['name']
        clsid = self.LABELS.index(name)
        gt_boxes = np.asarray([bbox[0], bbox[1], bbox[2], bbox[3], clsid],
                              dtype=np.float32)

        # Load image
        img_fn = '{}/{}'.format(self.img_dir, obj['filename'])
        img = cv.imread(img_fn).astype(np.float)
        img -= self.mean

        # Scaling
        im_size_min = np.min(img.shape[:2])
        im_size_max = np.max(img.shape[:2])
        im_scale = float(self.IMG_TARGET_SIZE) / float(im_size_min)

        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.IMG_MAX_SIZE:
            im_scale = float(self.IMG_MAX_SIZE) / float(im_size_max)
        img = cv.resize(img, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv.INTER_LINEAR)
        h, w = img.shape[:2]
        im_info = np.asarray([h, w, im_scale], dtype=np.float32)
        img = img.transpose(2, 0, 1).astype(np.float32)

        return img, im_info, gt_boxes


if __name__ == '__main__':
    dataset = VOC('data/VOCdevkit/VOC2007/JPEGImages',
                  'data/VOCdevkit/VOC2007/Annotations',
                  'data/VOCdevkit/VOC2007/ImageSets/Main', 'train')
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))

    dataset = VOC('data/VOCdevkit/VOC2007/JPEGImages',
                  'data/VOCdevkit/VOC2007/Annotations',
                  'data/VOCdevkit/VOC2007/ImageSets/Main', 'val')
    img, im_info, gt_boxes = dataset[0]
    print(img.shape)
    print(im_info)
    print(gt_boxes)
    print('len:', len(dataset))
