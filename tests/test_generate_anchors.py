#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
from PIL import Image
from PIL import ImageDraw

from models.generate_anchors import generate_anchors


class TestGenerateAnchors(unittest.TestCase):

    def test_generate_anchors(self):
        for bs, r, s in [[15, (0.5, 1, 2), (4, 8, 16, 32)],
                         [15, (0.5, 1, 2), (8, 16, 32)]]:
            anchors = generate_anchors(bs, r, s)
            canvas = Image.fromarray(np.zeros((1024, 1024), dtype='u1'))
            for rec in anchors:
                rec += 512
                draw = ImageDraw.Draw(canvas)
                x0, y0, x1, y1 = rec
                draw.rectangle([x0, y0, x1, y1], None, 255)
            canvas.save('tests/anchors_{}.png'.format('-'.join([str(a) for a in s])))
