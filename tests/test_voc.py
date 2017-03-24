#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import unittest
from lib.datasets.voc.dataset import VOC


class TestVOC(unittest.TestCase):

    def setUp(self):
        self.dataset = VOC('train', False)
        if not os.path.exists('tests/imgs'):
            os.makedirs('tests/imgs')

    def test_save_example(self):
        ret = self.dataset[:2]

        self.assertEqual(len(ret), 2)
        self.assertEqual(len(ret[0]), len(ret[1]))
        self.assertEqual(len(ret[0]), 3)

        for i in range(2):
            img, im_info, bbox = ret[i]
            img = img.transpose(1, 2, 0) + self.dataset.mean
            for bb in bbox:
                bb = [int(b) for b in bb]
                cv.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255))
            cv.imwrite('tests/imgs/example_{}.png'.format(i), img)
